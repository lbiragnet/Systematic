import sqlite3
import os
import glob
from datetime import datetime


def perform_backup(db_name, backup_folder="data/backups", max_backups=7):
    """
    Safely backs up the SQLite database and rotates old files.
    """

    # 1. Create backup directory if it doesn't exist
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
        print(f"Created backup directory: {backup_folder}")

    # 2. Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_filename = f"{os.path.splitext(db_name)[0]}_{timestamp}.db"
    backup_path = os.path.join(backup_folder, backup_filename)

    try:
        # 3. Perform the backup
        source_conn = sqlite3.connect(db_name)
        dest_conn = sqlite3.connect(backup_path)
        source_conn.backup(dest_conn)
        print(f"Backup successful: {backup_path}")
        dest_conn.close()
        source_conn.close()

        # 4. Cleanup / rotation (delete old backups)
        _rotate_backups(db_name, backup_folder, max_backups)

    except Exception as e:
        print(f"Backup failed: {e}")


def _rotate_backups(db_name, backup_folder, max_backups):
    """Internal function to delete the oldest files if we exceed the limit."""

    # Get a list of all backup files for this database
    base_name = os.path.splitext(db_name)[0]
    pattern = os.path.join(backup_folder, f"{base_name}_*.db")
    list_of_files = glob.glob(pattern)

    # Sort files by creation time (oldest first)
    list_of_files.sort(key=os.path.getctime)

    # Check if we have too many
    while len(list_of_files) > max_backups:
        file_to_remove = list_of_files.pop(0)  # Remove the first (oldest) item
        try:
            os.remove(file_to_remove)
            print(f"Rotated (deleted) old backup: {os.path.basename(file_to_remove)}")
        except OSError as e:
            print(f"Error deleting old backup {file_to_remove}: {e}")


if __name__ == "__main__":
    perform_backup()
