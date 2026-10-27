import argparse
import os
from pathlib import Path
import subprocess
import venv

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--venv', type=Path, default=ROOT / '.venv')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    environment = args.venv.expanduser().resolve()
    if not environment.exists():
        venv.create(environment, with_pip=True, system_site_packages=True)
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    command = [str(python), '-m', 'pip', 'install', '--no-build-isolation', '-e', str(ROOT)]
    if args.offline:
        command.append('--no-index')
    subprocess.run(command, check=True)
    print(f'Ready: {python} -m LRAEP_trainer self-test')


if __name__ == '__main__':
    main()
