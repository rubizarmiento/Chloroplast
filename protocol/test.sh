#Source function library
source /martini/rubiz/Github/Chloroplast/protocol/stacks_periodic.sh
wdir=/martini/rubiz/Github/Chloroplast

#---TESTING---
not_empty_surfaces=("suzanne_mesh")
odir=${wdir}/build_model_tests

#generate_plm -odir ${odir} -scale 40 -meshno 1 -new_dir false

#---EXECUTION---
#tsi_to_pdb 40


#delete_nan_coordinates suzanne_mesh.gro system_clean.gro

#---PCG--
write_pcg -odir ${odir} -str ${wdir}/build_model/ts2cg_files/input.str
#run_pcg
