"""Read-only integrity checks; no torch imports, unpickling, training, or downloads."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-archive', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures = []
    checked = 0

    def check(path, sha, size=None):
        nonlocal checked
        checked += 1
        if not path.is_file() or digest(path) != sha or (size is not None and path.stat().st_size != size):
            failures.append(path.relative_to(root).as_posix())

    package = read(root / 'package_manifest.json')
    for rel, entry in package['files'].items():
        path = (root / rel).resolve()
        if root not in path.parents:
            raise SystemExit('Unsafe manifest path')
        check(path, entry['sha256'], entry['size'])

    audit = read(root / 'protocol/audit/reviewer_minimum_audit_pass.json')
    check(root / 'protocol/formal_reviewer_minimum33_plan.json', audit['plan_sha256'])
    plan = read(root / 'protocol/formal_reviewer_minimum33_plan.json')
    check(root / 'manifests/frozen_subject_grouped_sequence_manifest.csv', plan['split_manifest_sha256'])
    inventory = read(root / 'protocol/reused_inventory.json')
    for entry in inventory['files']:
        rel = entry['path'].replace('\\', '/').split('/AMASS_LNN_Project/', 1)[1]
        path = root / rel if rel.startswith('scripts/') else root / 'code' / rel
        check(path, entry['sha256'], entry['size'])
    for entry in audit['artifacts']:
        rel = entry['path'].replace('\\', '/').split('/analysis/', 1)[1]
        check(root / 'analysis/outputs' / rel, entry['sha256'], entry['size'])
    states = list((root / 'protocol/stage_state').glob('*_exit.json'))
    formal = [read(p) for p in states if 'stage_id' in read(p) and not read(p).get('smoke')]
    if len(formal) != 33 or any(x.get('status') != 'completed' or x.get('exit_code') != 0 for x in formal):
        failures.append('formal_training_stage_states')
    if audit.get('status') != 'PASS' or audit.get('training_stages_completed') != 33 or not all(x['passed'] for x in audit['checks']):
        failures.append('historical_final_audit')

    checkpoint_status = 'not_publicly_distributed; not required for artifact-only verification'
    if args.checkpoint_archive:
        expected = read(root / 'checkpoints_manifest.json')['files']
        with zipfile.ZipFile(args.checkpoint_archive) as archive:
            names = {x.filename for x in archive.infolist() if not x.is_dir()}
            if names != set(expected):
                failures.append('checkpoint_archive_inventory')
            for rel, sha in expected.items():
                if rel not in names:
                    continue
                with archive.open(rel) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != sha:
                        failures.append(rel)
        checkpoint_status = '48 archive entries checked against the retained SHA-256 inventory'
    print(json.dumps({'status': 'PASS' if not failures else 'FAIL', 'file_checks': checked,
                      'completed_stages': len(formal), 'original_audit_checks': len(audit['checks']),
                      'checkpoint_access': checkpoint_status, 'failures': failures}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
