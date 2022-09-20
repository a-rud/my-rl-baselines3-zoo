import argparse
import sys

from synergy_curriculum.utils import get_and_create_save_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log-path", help="Path of logs to look for runs", type=str, required=True)
    parser.add_argument("-e", "--env", help="Name of environment.", type=str, required=True)
    args = parser.parse_args()

    save_path, folder_name = get_and_create_save_folder(directory=args.log_path, prefix=args.env)
    sys.stdout.write(f"{folder_name}")
