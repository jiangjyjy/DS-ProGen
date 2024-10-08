import foldcomp
from tqdm import tqdm

with foldcomp.open("/home/v-yantingli/mmp/foldcomp_data/afdb_rep_v4") as db:
  pbar = tqdm(total=len(db))
  for (name, pdb) in db:
    # save entries as seperate pdb files
    with open("afdb_v4/" + name.replace(".cif.gz", "") + ".pdb", "w") as f:
      f.write(pdb)
    pbar.update(1)
  pbar.close()