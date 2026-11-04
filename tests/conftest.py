# pytest loads this file automatically before running any tests in this folder.
# scPairtree's modules live in lib/ and bin/ and import each other by bare name
# (e.g. `import util`), so both folders must be on the import path.
import os
import sys

REPO_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(REPO_DIR, 'lib'))
sys.path.insert(0, os.path.join(REPO_DIR, 'bin'))
