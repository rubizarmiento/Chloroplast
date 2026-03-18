#Source function library
source /martini/rubiz/Github/Chloroplast/protocol/stacks_periodic.sh
wdir=/martini/rubiz/Github/Chloroplast
#---TESTING---
not_empty_surfaces=("suzanne_mesh")
#generate_plm arguments: scale - mesno - new_dir - adddomains
#generate_plm 40 1 false false

#---EXECUTION---
#tsi_to_pdb 40


#delete_nan_coordinates suzanne_mesh.gro system_clean.gro

#---PCG--
write_pcg ${wdir}/build_model/ts2cg_files/input.str
#run_pcg
