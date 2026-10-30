#!/usr/bin/env python3
"""One interface for pinned local inference profiles."""
import argparse
import json
import shlex
import subprocess
import sys
from lib import core, opencode, pi, swap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    swap_cmd = sub.add_parser('swap', help='Manage the optional two-model llama-swap coordinator')
    swap_sub = swap_cmd.add_subparsers(dest='swap_action', required=True)
    for action in ('install', 'config', 'start', 'status', 'stop', '_serve', '_cleanup'):
        command = swap_sub.add_parser(action)
        if action in ('config', 'start', '_serve'):
            command.add_argument('--port', type=int, default=1235)
            command.add_argument('--backend-port', type=int, default=1236)
            command.add_argument('--windows-backend-port', type=int, default=1257)
        if action in ('_serve', '_cleanup'):
            command.add_argument('--token', required=True)
        if action == '_serve':
            command.add_argument('--run-dir', required=True)
    for action in ('install', 'verify', 'start', 'stop', 'status', 'smoke'):
        cmd = sub.add_parser(action)
        cmd.add_argument('profile', choices=core.PROFILES, nargs='?' if action == 'status' else None)
        if action == 'start':
            cmd.add_argument('--dry-run', action='store_true')
            for key, cast in core.FIELDS.items():
                cmd.add_argument('--' + key.replace('_', '-'), type=cast)
    oc = sub.add_parser('opencode', help='Manage OpenCode model registration')
    oc_sub = oc.add_subparsers(dest='opencode_action', required=True)
    add = oc_sub.add_parser('add')
    add.add_argument('profile', choices=core.PROFILES)
    add.add_argument('--config', default='~/.config/opencode/opencode.jsonc')
    add.add_argument('--dry-run', action='store_true')
    add.add_argument('--update', action='store_true')
    for key in ('context', 'port', 'output-tokens'):
        add.add_argument('--' + key, type=int)
    pi_cmd = sub.add_parser('pi', help='Manage PI model registration')
    pi_sub = pi_cmd.add_subparsers(dest='pi_action', required=True)
    pi_add = pi_sub.add_parser('add')
    pi_add.add_argument('profile', choices=core.PROFILES)
    pi_add.add_argument('--config', default=pi.default_config())
    pi_add.add_argument('--dry-run', action='store_true')
    pi_add.add_argument('--update', action='store_true')
    for key in ('context', 'port', 'output-tokens'):
        pi_add.add_argument('--' + key, type=int)
    args = parser.parse_args()
    if args.action == 'swap':
        if args.swap_action == 'config':
            print(swap.config(args.port, args.backend_port, args.windows_backend_port), end='')
        elif args.swap_action == 'start':
            swap.start(args.port, args.backend_port, args.windows_backend_port)
        elif args.swap_action == '_serve':
            swap.serve(args.token, args.run_dir, args.port, args.backend_port,
                       args.windows_backend_port)
        elif args.swap_action == '_cleanup':
            swap.cleanup(args.token)
        else:
            getattr(swap, args.swap_action)()
        return
    if args.action == 'pi':
        pi.add(args.profile, args.config, args.dry_run, args.update,
               args.context, args.port, args.output_tokens)
        return
    if args.action == 'opencode':
        opencode.add(args.profile, args.config, args.dry_run, args.update,
                     args.context, args.port, args.output_tokens)
        return
    if args.action == 'status':
        for name in [args.profile] if args.profile else core.PROFILES:
            p = core.load(name)
            print(name, 'active' if core.scope_active(p) else 'inactive', 'port', p['port'])
        return
    if args.action == 'install':
        core.install(args.profile)
        return
    if args.action == 'verify':
        p = core.load(args.profile)
        root = core.artifact_root(p)
        for a in core.manifest('models')[args.profile]:
            core.verify(root / 'models' / a['file'], a)
        for a in core.manifest('runtimes')[p['runtime']]:
            core.verify(root / 'downloads' / a['file'], a)
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
