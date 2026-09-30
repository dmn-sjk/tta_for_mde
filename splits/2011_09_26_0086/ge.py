
base = "2011_09_26/2011_09_26_drive_0086_sync {:01d} l"
with open("val_files_bak.txt", 'w') as f:
    for i in range(706):
        f.write(base.format(i) + '\n')