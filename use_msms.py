import subprocess
import os

def generate_surface(pdb_file, output_dir):
    pdb_name = os.path.basename(pdb_file)
    prefix = pdb_name.split(".")[0]
    xyzr_file = os.path.join(output_dir, prefix + ".xyzr")
    pdb_to_xyzrn_command = f"pdb_to_xyzrn {pdb_file} > {xyzr_file}"
    subprocess.run(pdb_to_xyzrn_command, shell=True)
    msms_command = ["msms", "-if", xyzr_file, "-of", os.path.join(output_dir, prefix)]
    subprocess.run(msms_command, check=True)

# 使用示例
pdb_file = "data/test_data/2021-07-17_00000021_1_native.pdb"
output_dir = "/home/v-yantingli/mmp/surface_raw_data"
generate_surface(pdb_file, output_dir)