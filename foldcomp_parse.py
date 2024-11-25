import foldcomp
from tqdm import tqdm
import pickle

# with open('rep_name.pkl', 'rb') as f:
#   rep_name = pickle.load(f)

with foldcomp.open("foldcomp_data/afdb_swissprot_v4") as db:
  pbar = tqdm(total=len(db))
  for (name, pdb) in db:
    # save entries as seperate pdb files
    # if name.replace(".cif.gz", "") not in rep_name:
    with open("afdb_v4/" + name.replace(".cif.gz", "").replace(".pdb", "") + ".pdb", "w") as f:
      f.write(pdb)
    pbar.update(1)
  pbar.close()

# with open('lookup1.pkl', 'rb') as f:
#   ids = pickle.load(f)

# with foldcomp.open("mmp/foldcomp_data/highquality_clust30",ids=ids) as db:
#   pbar = tqdm(total=len(db))
#   for (name, pdb) in db:
#     # save entries as seperate pdb files
#     with open("afdb_v4/" + name.split(' ')[-1] + ".pdb", "w") as f:
#         f.write(pdb)
#     pbar.update(1)
#   pbar.close()