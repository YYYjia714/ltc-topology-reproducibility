"""Restore source import layout without downloading assets or running experiments."""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    destination = args.destination.resolve()
    if destination == source or source in destination.parents:
        raise SystemExit('Use a new directory outside the public package.')
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit('Destination must be absent or empty; existing experiments are protected.')
    destination.mkdir(parents=True, exist_ok=True)
    for p in (source / 'code').iterdir():
        if p.name == '__pycache__':
            continue
        if p.is_dir():
            shutil.copytree(p, destination / p.name)
        else:
            shutil.copy2(p, destination / p.name)
    for name in ('scripts', 'manifests', 'configs'):
        shutil.copytree(source / name, destination / name)
    shutil.copy2(source / 'requirements.txt', destination / 'requirements.txt')
    print('Source layout prepared. No dependencies installed, data copied, or experiments started.')


if __name__ == '__main__':
    main()
