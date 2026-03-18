#Source function library
source /martini/rubiz/thylakoid/scripts/stacks/stacks_periodic.sh

#---TESTING---
not_empty_surfaces=("suzanne_mesh")
#generate_plm arguments: scale - mesno - new_dir - adddomains
generate_plm 40 1 false false

#---STROMA LAMELLAE---
#not_empty_surfaces=("stroma_part2")
#not_empty_surfaces=("test3_oct26")
#not_empty_surfaces=(test4_oct30)
#not_empty_surface=("test3_oct26")

#---GRANA---
#not_empty_surfaces=("grana_part1")
#not_empty_surfaces=("grana_part2")
#not_empty_surfaces=("grana_part3")
#not_empty_surfaces=("grana_part4")

#---WHOLE---
#not_empty_surfaces=("inner_membrane_v2")

#not_empty_surfaces=("grana_v2")
#-------------------------------------#
#---EXECUTION---
#tsi_to_pdb 4000

#---PLM--
#PLM arguments: scale - mesno - new_dir - adddomains
#generate_plm 1 1 true false
#generate_plm 4000 1 true false

#delete_nan_coordinates system.gro system_clean.gro

#---PCG--
#write_pcg ../001/input.str
#run_pcg

#---NOT CONNECTED SURFACES--
#not_empty_surfaces=("grana_part1" "grana_part2" "grana_part3" "grana_part4" "stroma_part1" "stroma_part2")
#concatenate_surfaces "chloroplast"
#not_empty_surfaces=("chloroplast")
#cut_box_surfaces "chloroplast"

#---WHOLE--
#not_empty_surfaces=("inner_membrane_v1")
#cut_box_surfaces arguments: cutx cuty cutz cut_x cut_y cut_z
#cut_box_surfaces "176 110 0 960 110 0"




