"""Decompress video into image frames."""

import argparse
import glob
import multiprocessing as mp
import os
import shutil
from functools import partial

import cv2
import h5py
import numpy as np
import tqdm
import logging
import zipfile
import io
import tarfile

if cv2.__version__[0] != "4":
    print("Please upgrade your OpenCV package to 4.x.")
    exit(1)

VIEWS = [
    ("front", "Front"),
    ("left_45", "Left 45°"),
    ("left_90", "Left 90°"),
    ("right_45", "Right 45°"),
    ("right_90", "Right 90°"),
    ("left_stereo", "Front (Stereo)"),
    ("center", "Center (for LiDAR)"),
]

DATA_GROUPS = [
    ("img", "zip", "RGB Image"),
    ("det_2d", "json", "2D Detection and Tracking"),
    ("det_3d", "json", "3D Detection and Tracking"),
    ("semseg", "zip", "Semantic Segmentation"),
    ("det_insseg_2d", "json", "Instance Segmentation"),
    ("flow", "zip", "Optical Flow"),
    ("depth", "zip", "Depth Maps (24-bit)"),
    ("depth_8bit", "zip", "Depth Maps (8-bit)"),
    ("seq", "csv", "Sequence Info"),
    ("lidar", "zip", "LiDAR Point Cloud"),
]

class LoggerSingleton:
    __instance = None

    @staticmethod
    def get_logger():
        if LoggerSingleton.__instance is None:
            LoggerSingleton()
        return LoggerSingleton.__instance.logger

    def __init__(self):
        if LoggerSingleton.__instance is not None:
            raise Exception(
                "LoggerSingleton is a singleton class, use get_logger() instead"
            )
        else:
            self.logger = logging.getLogger("shift_dev_logger")
            self.logger.setLevel(logging.DEBUG)
            log_formatter = logging.Formatter(
                "[%(asctime)s] SHIFT DevKit - %(levelname)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
            )
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(log_formatter)
            self.logger.addHandler(ch)
            self.logger.propagate = False
            
            LoggerSingleton.__instance = self


def setup_logger():
    logger = LoggerSingleton.get_logger()
    return logger


class ArchiveWriter:
    def __init__(self, filename) -> None:
        self.filename = filename

    def get_list(self):
        raise NotImplementedError

    def add_file(self, name, arcname):
        raise NotImplementedError

    def closs(self):
        raise NotImplementedError

class ArchiveReader:
    def __init__(self, filename) -> None:
        self.filename = filename

    def get_list(self):
        raise NotImplementedError

    def get_file(self, name):
        raise NotImplementedError

    def closs(self):
        raise NotImplementedError

class ZipArchiveWriter(ArchiveWriter):
    default_ext = "zip"

    def __init__(self, filename) -> None:
        super().__init__(filename)
        self.file = zipfile.ZipFile(filename, "w")

    def add_file(self, name, arcname="."):
        for root, dirs, files in os.walk(name):
            for file in files:
                filepath = os.path.join(root, file)
                self.file.write(filepath, os.path.join(arcname, file))

    def get_list(self):
        return self.file.namelist()

    def close(self):
        self.file.close()


class TarArchiveReader(ArchiveReader):
    def __init__(self, filename) -> None:
        super().__init__(filename)
        self.file = tarfile.TarFile(filename, "r")

    def get_file(self, name):
        data = self.file.extractfile(name)
        bytes_io = io.BytesIO(data)
        return bytes_io

    def extract_file(self, name, output_dir):
        self.file.extract(name, output_dir)

    def get_list(self):
        return self.file.getnames()

    def close(self):
        self.file.close()


class TarArchiveWriter(ArchiveWriter):
    default_ext = "tar"

    def __init__(self, filename) -> None:
        self.filename = filename
        self.file = tarfile.TarFile(filename, "w")
        # print(f"Loaded {filename}.")

    def add_file(self, name, arcname):
        self.file.add(name, arcname=arcname)

    def get_list(self):
        return self.file.getnames()

    def close(self):
        self.file.close()


DATA_GROUP_NAMES = [item[0] for item in DATA_GROUPS]
VIEW_NAMES = [item[0] for item in VIEWS]


def get_suffix(tar_file):
    filepath, filename = os.path.split(tar_file.filename)
    group_name = os.path.splitext(filename)[0]
    view_name = os.path.split(filepath)[1]
    assert (view_name in VIEW_NAMES) and (
        group_name in DATA_GROUP_NAMES
    ), f"It seems that {filename} doesn't follow the dataset structure."
    if group_name == "img":
        ext = "jpg"
    else:
        ext = "png"
    return view_name, group_name, ext


def extract_video(tar_file, video_name, output_dir, tmp_dir):
    view_name, group_name, ext = get_suffix(tar_file)
    tar_file.extract_file(video_name, tmp_dir)
    video = cv2.VideoCapture(os.path.join(tmp_dir, video_name))
    if not video.isOpened():
        logger.error("Error opening video stream or file!")
    frame_id = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    "{:08d}_{}_{}.{}".format(frame_id, group_name, view_name, ext),
                ),
                frame,
            )
            frame_id += 1
        else:
            break
    video.release()
    os.remove(os.path.join(tmp_dir, video_name))


def convert_to_archive(
    tar_filepath, tmp_dir, show_progress_bar=False, writer=TarArchiveWriter
):
    try:
        tar_file = TarArchiveReader(tar_filepath)
    except Exception as e:
        logger.error("Cannot open {}. ".format(tar_filepath) + e)
        return
    try:
        out_filepath = tar_filepath.replace(
            ".tar", f"_decompressed.{writer.default_ext}"
        )
        archive_writer = writer(out_filepath)
    except Exception as e:
        logger.error("Cannot create {}. ".format(out_filepath) + e)
        return

    file_list = tar_file.get_list()
    if show_progress_bar:
        file_list = tqdm.tqdm(file_list)
    for f in file_list:
        if f.endswith(".mp4"):
            output_dir = os.path.join(
                tar_filepath.replace(".tar", "_tmp"),
                os.path.basename(f).split(".")[0],
            )
            os.makedirs(output_dir, exist_ok=True)
            extract_video(tar_file, f, output_dir, tmp_dir)
            archive_writer.add_file(
                output_dir,
                arcname=os.path.basename(f).split(".")[0],
            )
            shutil.rmtree(output_dir)
    tar_file.close()
    archive_writer.close()


def convert_to_hdf5(tar_filepath, tmp_dir, show_progress_bar=False):
    try:
        tar_file = TarArchiveReader(tar_filepath)
    except Exception as e:
        logger.error("Cannot open {}. ".format(tar_filepath) + e)
        return
    try:
        hdf5_filepath = tar_filepath.replace(".tar", "_decompressed.hdf5")
        hdf5_file = h5py.File(hdf5_filepath, mode="w")
    except Exception as e:
        logger.error("Cannot create {}. ".format(hdf5_filepath) + e)
        return

    def write_to_hdf5(seq, folder_path):
        for f in os.listdir(folder_path):
            if seq in hdf5_file:
                g = hdf5_file[seq]
            else:
                g = hdf5_file.create_group(seq)
            with open(os.path.join(folder_path, f), "rb") as fp:
                file_content = fp.read()
                g.create_dataset(f, data=np.frombuffer(file_content, dtype="uint8"))

    file_list = tar_file.get_list()
    if show_progress_bar:
        file_list = tqdm.tqdm(file_list)
    for f in file_list:
        if f.endswith(".mp4"):
            output_dir = os.path.join(
                tar_filepath.replace(".tar", "_tmp"),
                os.path.basename(f).split(".")[0],
            )
            os.makedirs(output_dir, exist_ok=True)
            extract_video(tar_file, f, output_dir, tmp_dir)
            write_to_hdf5(os.path.splitext(f)[0], output_dir)
            shutil.rmtree(output_dir)
    tar_file.close()
    hdf5_file.close()


def convert_to_folder(tar_filepath, tmp_dir, show_progress_bar=False):
    try:
        tar_file = TarArchiveReader(tar_filepath)
    except Exception as e:
        logger.error("Cannot open {}. ".format(tar_filepath) + e)
        return

    file_list = tar_file.get_list()
    if show_progress_bar:
        file_list = tqdm.tqdm(file_list)
    for f in file_list:
        if f.endswith(".mp4"):
            output_dir = os.path.join(
                tar_filepath.replace(".tar", ""), f.replace(".mp4", "")
            )
            os.makedirs(output_dir, exist_ok=True)
            extract_video(tar_file, f, output_dir, tmp_dir)


CONVERT_MAP = dict(
    folder=convert_to_folder,
    tar=partial(convert_to_archive, writer=TarArchiveWriter),
    zip=partial(convert_to_archive, writer=ZipArchiveWriter),
    hdf5=convert_to_hdf5,
)


def main():
    parser = argparse.ArgumentParser(
        description="Decompress tar files of videos into image frames."
    )
    parser.add_argument("files", type=str, help="File pattern to match tar files.")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="folder",
        choices=["folder", "tar", "zip", "hdf5"],
        help="Conversion mode. Defines the type of output.",
    )
    parser.add_argument(
        "-j", "--jobs", default=1, type=int, help="Number of jobs to run in parallel."
    )
    parser.add_argument(
        "--tmp_dir",
        default="/tmp/shift-dataset/",
        help="Temporary folder for decompressed video files.",
    )
    args = parser.parse_args()

    if args.files[-4:] != ".tar":
        logger.error("File pattern must end with '.tar'!")
        exit()

    files = []
    for file in glob.glob(args.files, recursive=True):
        if file.endswith("_decompressed.tar"):
            logger.warning(f"Skip a decompressed tar file: {file}.")
        else:
            files.append(file)

    os.makedirs(args.tmp_dir, exist_ok=True)
    logger.info("Files to convert: " + str(len(files)))
    logger.info(f"Starting conversion to {args.mode}")
    convert = CONVERT_MAP[args.mode]

    if args.jobs > 1:
        convert_fn = partial(convert, tmp_dir=args.tmp_dir)
        with mp.Pool(args.jobs) as pool:
            _ = list(tqdm.tqdm(pool.imap(convert_fn, files), total=len(files)))
    else:
        logger.info(
            "Note: You can also run this code using multi-processing by setting `-j` option."
        )
        for f in files:
            logger.info("Processing " + f)
            convert(f, args.tmp_dir, show_progress_bar=True)


if __name__ == "__main__":
    logger = setup_logger()
    main()
