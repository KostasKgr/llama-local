#!/usr/bin/env python3
"""One interface for pinned local inference profiles."""
import argparse
import json
import shlex
import subprocess
import sys
from lib import core, opencode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('install', 'verify', 'start', 'stop', 'status', 'smoke'):
        cmd = sub.add_parser(action)
        cmd.add_argument('profile', choices=('swift15', 'qwen36'), nargs='?' if action == 'status' else None)
        if action == 'start':
            cmd.add_argument('--dry-run', action='store_true')
            for key, cast in core.FIELDS.items():
                cmd.add_argument('--' + key.replace('_', '-'), type=cast)
    oc = sub.add_parser('opencode', help='Manage OpenCode model registration')
    oc_sub = oc.add_subparsers(dest='opencode_action', required=True)
    add = oc_sub.add_parser('add')
    add.add_argument('profile', choices=('swift15', 'qwen36'))
    add.add_argument('--config', default='~/.config/opencode/opencode.jsonc')
    add.add_argument('--dry-run', action='store_true')
    add.add_argument('--update', action='store_true')
    for key in ('context', 'port', 'output-tokens'):
        add.add_argument('--' + key, type=int)
    args = parser.parse_args()
    if args.action == 'opencode':
        opencode.add(args.profile, args.config, args.dry_run, args.update,
                     args.context, args.port, args.output_tokens)
        return
    if args.action == 'status':
        for name in [args.profile] if args.profile else ['swift15', 'qwen36']:
            p = core.load(name)
            print(name, 'active' if core.scope_active(p) else 'inactive', 'port', p['port'])
        return
    if args.action == 'install':
        core.install(args.profile)
        return
    if args.action == 'verify':
        p = core.load(args.profile)
        for a in core.manifest('models')[args.profile]:
            core.verify(core.ROOT / 'models' / a['file'], a)
        for a in core.manifest('runtimes')[p['runtime']]:
            core.verify(core.ROOT / 'downloads' / a['file'], a)
        print('All pinned artifacts verified')
        return
    if args.action == 'smoke':
        subprocess.run([sys.executable, str(core.ROOT / 'experiments/smoke.py'), args.profile], check=True)
        return
    p = core.load(args.profile, {k: getattr(args, k, None) for k in core.FIELDS})
    if args.action == 'stop':
        core.stop(p)
    elif args.dry_run:
        print(json.dumps(p, indent=2))
        print(shlex.join(core.command(p)))
    else:
        core.start(p)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
