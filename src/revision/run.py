"""Последовательный запуск. При ошибке останавливаемся и сохраняем причину."""
import argparse
import importlib.metadata
import platform
import subprocess
import sys
import traceback
from .common import ROOT, RUN, config, save_json, status, checkpoint_backup


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start', choices=['prepare','features','train','review'],default='prepare')
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    save_json(RUN/'environment.json',dict(python=sys.version,platform=platform.platform(),
              packages={p.metadata['Name']:p.version for p in importlib.metadata.distributions()},config=config()))
    stages=['prepare','features','train','review']
    try:
        for stage in stages[stages.index(args.start):]:
            status(stage+'_starting')
            subprocess.run([sys.executable,'-u','-m','src.revision.'+stage],cwd=ROOT,check=True)
            checkpoint_backup()
        status('complete',message='Training and review artifacts ready. Human review still required.')
        checkpoint_backup()
    except Exception as error:
        status('failed',error=str(error),traceback=traceback.format_exc())
        checkpoint_backup()
        raise


if __name__=='__main__':
    main()
