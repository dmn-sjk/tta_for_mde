
base = "2011_09_26/2011_09_26_drive_0117_sync {:01d} l"
with open("val_files_bak.txt", 'w') as f:
    for i in range(660):
        f.write(base.format(i) + '\n')