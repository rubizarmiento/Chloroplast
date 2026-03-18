scripts=/martini/rubiz/thylakoid/scripts
wdir=/martini/rubiz/Github/Chloroplast
#TS2CG=/martini/rubiz/stacks/TS2CG1.2
TS2CG=/martini/rubiz/stacks/TS2CG1.1-master/TS2CG1.1.1.0_r
objs=${wdir}/objs

#region main functions
tsi_to_pdb(){
    scale=${1}
    cd ${wdir}
    length=${#not_empty_surfaces[@]}
    for i in $(seq 1 ${length}); do
        case=${not_empty_surfaces[$i-1]}
        cd ${wdir}
        mkdir -p ${case}    
        cd ${case}
        python3 ${scripts}/obj_to_tsi.py -i ${objs}/${case}.obj -o ${case}.tsi -scale ${scale} 
        python3 ${scripts}/tsi_to_pdb.py -f ${wdir}/${case}/${case}.tsi -o ${wdir}/${case}/${case}.gro 
    done
}

generate_plm(){
    local odir="" scale="" Mashno="" new_dir="false" material="false"
    while [[ $# -gt 0 ]]; do
        case $1 in
            -odir)     odir="$2";     shift 2;;
            -scale)    scale="$2";    shift 2;;
            -meshno)   Mashno="$2";   shift 2;;
            -new_dir)  new_dir="$2";  shift 2;;
            -material) material="$2"; shift 2;;
            *) echo "Unknown argument: $1"; return 1;;
        esac
    done
    cd ${odir}
    echo "Generating PLM files for ${not_empty_surfaces[@]}"
    length=${#not_empty_surfaces[@]}
    for i in $(seq 1 ${length}); do
        case=${not_empty_surfaces[$i-1]}
        if ${new_dir}; then
            rm -r ${odir}/${case}
            mkdir -p ${odir}/${case}   
        fi
 
        cd ${odir}/${case}
        echo "---Step1: Generating TSI file for ${case}---"
        if ${material};then
            python3 ${scripts}/obj_to_tsi.py -i ${objs}/${case}.obj -o ${case}.tsi -scale ${scale} -box "2000" "2000" "2000" -material 
        else
            python3 ${scripts}/obj_to_tsi.py -i ${objs}/${case}.obj -o ${case}.tsi -scale ${scale} -box "2000" "2000" "2000" 
        fi
        
        rm -rf ${odir}/pointvisualization_data
        rm -rf ${odir}/point
        rm -f ${odir}/extended.tsi
        echo "---Step2: Generating PLM files for ${case}---"
        echo "${TS2CG}/PLM -TSfile ${case}.tsi -bilayerThickness 3 -Mashno ${Mashno} -smooth -AlgType Type1" >> plm.sh
        ${TS2CG}/PLM -TSfile ${case}.tsi -bilayerThickness 3 -Mashno ${Mashno} -smooth -AlgType Type1 > error.log
        cat error.log
        echo "Poblematic triangles:"
        bash ${scripts}/column_to_array.sh error.log > bad_triangles.txt
        cat bad_triangles.txt

    done
    echo "Output directory: ${odir}"
}

generate_plm_inclusions(){
    scale=${1}
    Mashno=${2}
    new_dir=${3}
    inclusions_file=${4}
    cd ${wdir}
    echo "Generating PLM files for ${not_empty_surfaces[@]}"
    length=${#not_empty_surfaces[@]}
    for i in $(seq 1 ${length}); do
        case=${not_empty_surfaces[$i-1]}
        if ${new_dir}; then
            echo "New directory"
            rm -r ${wdir}/${case}
            mkdir -p ${wdir}/${case}   
        fi
 
        cd ${wdir}/${case}
        echo "Step1: Generating TSI file for ${case}/n"

        cp /martini/rubiz/stacks/stacks_popc/str/proteins_v1.str .
        cp /martini/rubiz/stacks/stacks_popc/str/${inclusions_file} .

        python3 ${scripts}/obj_to_tsi.py -i ${objs}/${case}.obj -o ${case}.tsi -scale ${scale} -box "2000" "2000" "2000" 
        
        cat ${inclusions_file} >> ${case}.tsi
        echo "Added ${inclusions_file} to ${case}.tsi"

        rm -r pointvisualization_data
        rm -r point
        rm extended.tsi
        echo "Step2: Generating PLM files for ${case}\n"
        ${TS2CG}/PLM -TSfile ${case}.tsi -bilayerThickness 3 -Mashno ${Mashno} -smooth > error.log
        cat error.log
        echo "Poblematic triangles:"
        bash ${scripts}/column_to_array.sh error.log > bad_triangles.txt
        cat bad_triangles.txt

    done
}


write_pcg(){
    local odir="" str=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            -odir) odir="$2"; shift 2;;
            -str)  str="$2";  shift 2;;
            *) echo "Unknown argument: $1"; return 1;;
        esac
    done
    output="run_pcg.sh"
    length=${#not_empty_surfaces[@]}
    for i in $(seq 1 ${length}); do
        case=${not_empty_surfaces[$i-1]}
        cd ${odir}
        output=${odir}/run_pcg.sh
        #echo "${TS2CG}/PLM -TSfile extended.tsi -bilayerThickness 3 -rescalefactor 1 1 1 -Mashno 1 -o 2" > ${output}
        #echo "${TS2CG}/PLM -TSfile extended.tsi -bilayerThickness 3 -rescalefactor 1 1 1 -Mashno 1 -o 3" >> ${output}
        echo "${TS2CG}/PCG -str ${str} -Bondlength 0.1 -LLIB ${wdir}/build_model/ts2cg_files/Martini3.LIB -defout system -dts point" > ${output}
        echo "${odir}/run_pcg.sh was written"
    done
}

run_pcg(){
    local odir="" str=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            -odir) odir="$2"; shift 2;;
            -str)  str="$2";  shift 2;;
            *) echo "Unknown argument: $1"; return 1;;
        esac
    done

    length=${#not_empty_surfaces[@]}
    for i in $(seq 1 ${length}); do
        case=${not_empty_surfaces[$i-1]}
        cd ${odir}
        rm -f system.gro system.top
        bash ${odir}/run_pcg.sh
    done
}

#endregion

#Function to delete nan coordinates in an structure file
delete_nan_coordinates(){
    length=${#not_empty_surfaces[@]}
    for i in $(seq 1 ${length}); do
        case=${not_empty_surfaces[$i-1]}
        cd ${wdir}/${case}
        structure_file=${1}
        output=${2}
        echo "Deleting nan coordinates from ${structure_file}"
        n_nan=$(grep -c "nan" ${structure_file})
        n_atoms_initial=$(head -n 2 ${structure_file} | tail -n 1)
        n_atoms_final=$((${n_atoms_initial}-${n_nan}))
        echo "Number of atoms in the output file: ${n_atoms_final}"
        echo "Number of nan coordinates: ${n_nan}"
        grep -v "nan" ${structure_file} > ${output}
        sed -i "2s/.*/${n_atoms_final}/" ${output}

        #Renumber the atoms
        gmx genconf -f ${output} -o ${output} -renumber
        echo "Output file: ${output}"
    done
}

split_structue(){
    structure_file=${1}
    n=${2}
    #Bonx size is the last line
    box_size = $(tail -n 1 ${structure_file})
    #Split the structure file in n parts
    split -l ${n} ${structure_file} ${structure_file}_part
    #For the first file, delete first two lines
    sed -i '1,2d' ${structure_file}_partaa
    #For the last file, delete the last line
    sed -i '$d' ${structure_file}_partab
    for i in $(seq 1 ${n}); do
        n_lines=$(wc -l ${structure_file}_part${i} | awk '{print $1}')
        #Add comment as first line and number of atoms as second line
        sed -i "1s/.*/${structure_file}_part${i}/" ${structure_file}_part${i}
        sed -i "2s/.*/${n_lines}/" ${structure_file}_part${i}
        #Add the box size to the last line
        sed -i '$a\'${box_size} ${structure_file}_part${i}
    done
}

split_by_resname(){
    structure_file=${1}
    resnames_array=(${2})
    n_resnames=${#resnames_array[@]}
    box_size=$(tail -n 1 ${structure_file})
    for i in $(seq 1 ${n_resnames}); do
        echo "Splitting ${structure_file} by ${resnames_array[$i-1]}"
        resname=${resnames_array[$i-1]}
        grep "${resname}" ${structure_file} > ${resname}.gro
        n_lines=$(wc -l ${resname}.gro | awk '{print $1}')
        n_lines=$((${n_lines}-2))
        #Add comment as first line and number of atoms as second line
        sed -i "1s/.*/${resname}/" ${resname}.gro
        sed -i "2s/.*/${n_lines}/" ${resname}.gro
        #Add the box size to the last line
        echo ${box_size} >> ${resname}.gro
        gmx genconf -f ${resname}.gro -o ${resname}.gro -renumber
    done
}

#region main not_used_functions

check_tests(){
    n_tests=36
    rm -f ${wdir}/test_results.txt
    for i in $(seq 1 ${n_tests}); do
        if [[ " ${empty_surfaces[@]} " =~ " ${i} " ]]; then
            continue
        else
            case=$(printf "%03d" ${i})
            file=${wdir}/${case}/pointvisualization_data/Upper.gro
            if [ ! -s ${file} ]; then
                echo "Test failed: ${file} is empty" >> ${wdir}/test_results.txt
            else 
                echo "Test passed: ${file} is not empty" >> ${wdir}/test_results.txt
            fi
        fi
    done
    cat ${wdir}/test_results.txt
}

tight_box(){
    file=${1}
    output=${2}
    echo "---SETTING BOX TIGHT FOR ${PWD}/${file}---"
    python3 ${scripts}/extract_structure.py -f ${file} -o ${output} -tight -sel "all"
}

cut_box(){
    file=${1}
    output=${2}
    x=${3}
    y=${4}
    z=${5}
    echo ""
    echo "---CUTTING BOX FOR ${PWD}/${file}---"
    python3 ${scripts}/extract_structure.py -f ${file} -o ${output} -cutx ${x} -cuty ${y} -cutz ${z} -sel "all" -whole
}

cut_box_tight(){
    file=${1}
    output=${2}
    x=${3}
    y=${4}
    z=${5}
    x_negative=${6}
    y_negative=${7}
    z_negative=${8}
    echo "---CUTTING BOX FOR ${PWD}/${file}---"
    #python3 ${scripts}/extract_structure.py -f ${file} -o ${output} -p cut.top -cutx ${x} -cuty ${y} -cutz ${z} -cut_x ${x_negative} -cut_y ${y_negative} -cut_z ${z_negative} -sel "all" -tight -whole
    python3 ${scripts}/extract_structure.py -f ${file} -o ${output} -cutx ${x} -cuty ${y} -cutz ${z} -cut_x ${x_negative} -cut_y ${y_negative} -cut_z ${z_negative} -sel "all" -tight
}

cut_box_surfaces(){
    arguments_array=(${1})
    length=${#not_empty_surfaces[@]}
    for i in $(seq 1 ${length}); do
        case=${not_empty_surfaces[$i-1]}
        cd ${wdir}/${case}
        cut_box_tight system.gro tight_box.gro ${arguments_array[@]}
    done
}

concatenate_universe_from_testfile(){
    name=${1}
    test_file=${2}
    rm -f ${wdir}/${name}.gro
    #Needs to be run after check_tests
    n_tests=36
    cat ${wdir}/${test_file}
    upper=$(grep "not empty" ${wdir}/${test_file} | awk '{print $3}')
    echo ${upper}
    python3 ${scripts}/concatenate_universe.py -f ${upper} -o ${wdir}/${name}.gro -box "4000" "4000" "4000" 
}

concatenate_universe(){
    output=${1} 
    x=${3}
    y=${4}
    z=${5}
    rm -f ${output}.gro
    echo ${files[@]}
    python3 ${scripts}/concatenate_universe.py -f ${files} -o ${output}.gro -box ${x} ${y} ${z}
}

concatenate_surfaces(){
    output=${1}
    paths=()
    length=${#not_empty_surfaces[@]}
    for i in $(seq 1 ${length}); do
        case=${not_empty_surfaces[$i-1]}
        paths+=("${wdir}/${case}/system.gro")
    done
    output_dir=${wdir}/${output}
    mkdir -p ${output_dir}
    output_file=${output_dir}/system.gro
    echo "Concatenating ${paths[@]} to ${output_file}"
    python3 ${scripts}/concatenate_universe.py -f ${paths[@]} -o ${output_file} -box 2000 2000 2000
}


concatenate_parts(){
    name=${1}
    test_file=${2}
    id=${3}
    rm -f ${wdir}/${name}
    upper=$(grep "not empty" ${wdir}/${test_file} | awk '{print $3}')
    rm -f ${wdir}/${name}
    python3 ${scripts}/concatenate_universe.py -f ${upper} -o ${wdir}/${name}.gro
    gmx editconf -f ${wdir}/${name}.gro -o ${wdir}/${name}_c.gro 
    echo "Cancatenated gro: ${wdir}/${name}_c.gro"
}

generate_pcggenerate_pcg(){
    cd ${wdir}
    n_tests=36
    for i in $(seq 1 ${n_tests}); do
        if [[ " ${empty_surfaces[@]} " =~ " ${i} " ]]; then
            continue
        else
            case=$(printf "%03d" ${i})
            mkdir -p ${wdir}/${case}
            cd ${wdir}/${case}
            scale=3333
            rescalefactor="1 1 1"
            Mashno=4
            file=${wdir}/${case}/pointvisualization_data/Upper.gro
            rm -f ${file}
            python3 ${scripts}/obj_to_tsi.py -i ${objs}/FigS7_3D.${case}.obj -o ${case}.tsi -scale ${scale} -box "200" "200" "200" 
            ${TS2CG}/PLM -TSfile ${case}.tsi -bilayerThickness 3 -rescalefactor ${rescalefactor} -Mashno ${Mashno} -smooth
            ##Check that the file /point/OuterBM.dat is not empty
            file=${wdir}/${case}/pointvisualization_data/Upper.gro
            if [ ! -s ${file} ]; then
                echo "Test failed: ${file} is empty"
                #Write ${output} to reproduce the test
                echo "Debug with ${wdir}/${case}/run_plm.sh"
                write_plm ${wdir}/${case}/run_plm.sh
            else 
                python3 ${scripts}/universe_to_png.py -f ${file} -o ${wdir}/${case}/pointvisualization_data/Upper.png
                #code ${wdir}/${case}/pointvisualization_data/Upper.png
                echo "Test passed: ${file} is not empty"
                #Write ${output} to reproduce the test
                write_plm ${wdir}/${case}/run_plm.sh
            fi
            cd ${wdir}
        fi
    done
}


check_pcg(){
    n_tests=36
    rm -f ${wdir}/test_pcg.txt
    for i in $(seq 1 ${n_tests}); do
        if [[ " ${empty_surfaces[@]} " =~ " ${i} " ]]; then
            continue
        else
            case=$(printf "%03d" ${i})
            file=${wdir}/${case}/system.gro
            if [ ! -s ${file} ]; then
                echo "Test failed: ${file} is empty" >> ${wdir}/test_pcg.txt
            else 
                echo "Test passed: ${file} is not empty" >> ${wdir}/test_pcg.txt
            fi
        fi
    done
    cat ${wdir}/test_results.txt
}

write_thylakoid_itps(){
    top=${1}
    itps_dir=/martini/rubiz/thylakoid/templates/itps
    version="alpha"
    echo "#include \"${itps_dir}/martini_v3.0.0.itp\"" > struct.temp
    echo "#include \"${itps_dir}/version_nov22/martini_v3.0_ffbonded_alpha_v1.itp\"" >> struct.temp
    echo "#include \"${itps_dir}/martini_v3.0.0_ions_v1.itp\"" >> struct.temp
    echo "#include \"${itps_dir}/martini_v3.0.0_solvents_v1.itp\""  >> struct.temp
    echo "#include \"martini_v3.0_diacerolglycolipids_dev_v${version}.itp\"" >> struct.temp
    echo "#include \"martini_v3.0_phospholipids_PG_dev_v${version}.itp\"" >> struct.temp



    echo "" >> struct.temp
    echo "---${top}---"
    echo ""
    cp ${top} ${top}.backup
    cat struct.temp ${top}.backup > ${top}
    rm struct.temp
    cat ${top}

}

gen_ndx_stacks_mda(){
    input=${1}
    o=${2}
    selection_array=("all" "name GL1" "not name GL1")
    names_array=("System" "FLEXIBLE" "POSRES")

    python3 ${scripts}/sel_to_ndx.py -f ${input} -o ${o} -s "${selection_array[@]}" -n "${names_array[@]}"

}

gen_ndx_stacks(){
    input=${1}
    o=${2}
    selection_array=("a GL1" "!a GL1")

    output_array=("!GL1" " GL1")
    names_array=("FLEXIBLE" "POSRES")

    #Check if the arrays have the same length
    if [ ${#selection_array[@]} -ne ${#names_array[@]} ]; then
        echo "ERROR: The arrays have different lengths"
        exit 1
    fi

    command_array=()
    for i in $(seq 1 ${#selection_array[@]}); do
        command_array+=("${selection_array[$i-1]}")
        command_array+=("\n")
    done
    echo -e "${command_array[@]}\nq" | gmx make_ndx -f ${input} -o ${o} #Currently, gmx is faster in reading the universe

    #Replace output_array with names_array
    #Scape ! character
    for i in $(seq 1 ${#output_array[@]}); do
        output_array[$i-1]=$(echo ${output_array[$i-1]} | sed 's/!/\\!/g')
    done


    for i in $(seq 1 ${#output_array[@]}); do
        sed -i "s/${output_array[$i-1]}/${names_array[$i-1]}/g" ${o}
    done

    echo "Index file ${o} was written"
    echo "Groups in ${o}:"
    grep "\[" ${o}

}

write_min(){
    #SOFTCORE POTENTIAL MINIMIZATION
    output_run_file=${1}
    name=${2}
    mdp=${3}
    gro=${4}
    top=${5}
    mdps_dir=/martini/rubiz/thylakoid/templates/mdps
    echo "Minimizing ${gro} to ${output_gro} with ${mdp}"
    echo "cp ${mdps_dir}/${mdp} ." >> ${output_run_file}
    echo "" >> ${output_run_file}
    echo "gmx grompp -f ${mdp} -c ${gro} -p ${top} -o ${name} -maxwarn 2" >> ${output_run_file}
    echo "gmx mdrun -v -deffnm ${name} -ntmpi 1" >> ${output_run_file}
    echo "" >> ${output_run_file}
}

write_min_restraints(){
    #SOFTCORE POTENTIAL MINIMIZATION
    output_run_file=${1}
    name=${2}
    mdp=${3}
    gro=${4}
    top=${5}
    mdps_dir=/martini/rubiz/thylakoid/templates/mdps
    echo "Minimizing ${gro} to ${output_gro} with ${mdp}"
    echo "cp ${mdps_dir}/${mdp} ." >> ${output_run_file}
    echo "" >> ${output_run_file}
    echo "gmx grompp -f ${mdp} -c ${gro} -p ${top} -r ${gro} -o ${name} -maxwarn 2" >> ${output_run_file}
    echo "gmx mdrun -v -deffnm ${name} -ntmpi 1" >> ${output_run_file}
    echo "" >> ${output_run_file}
}


copy_restraints(){
    suffix=${1}
    itps_dir=/martini/rubiz/thylakoid/templates/itps/version_nov22
    cp ${itps_dir}/posre_pg_${suffix}.itp posre_pg.itp
    cp ${itps_dir}/posre_mg_${suffix}.itp posre_mg.itp
    cp ${itps_dir}/posre_gg_${suffix}.itp posre_gg.itp
    cp ${itps_dir}/posre_sq_${suffix}.itp posre_sq.itp
    echo "Restraints were copied"
}

write_stacks_run(){
    rm -f run.sh
    write_min_restraints "run.sh" "1_1_em_softcore" "em_soft_core.mdp" "thight_box.gro" system.top 
    write_min "run.sh" "1_2_em" "min.mdp" "1_2_em_softcore.gro" system.top 
    echo "---run.sh---"
    cat run.sh
}

write_glycolipids_run(){
    rm -f run.sh
    write_min_restraints "run.sh" "1_1_em_softcore" "em_soft_core.mdp" "system.gro" system.top 
    write_min "run.sh" "1_2_em" "min.mdp" "1_2_em_softcore.gro" system.top 
    echo "---run.sh---"
    cat run.sh
}

generate_insane(){
    molecule=${1}
    x=8
    y=8
    dz=15
    insane_command="-l ${molecule}:1"

    python2 ${scripts}/insane_M3_lipids.py -x ${x} -y ${y} -dz ${dz} -sol W ${insane_command} -o system.gro  -p system.top -salt 0 

    #Delete first line of system.top
    sed -i '1d' system.top
    cat system.top
}

test_glycolipids(){
    molecule=${1}
    generate_insane ${molecule}
    version=alpha
    cp /martini/rubiz/thylakoid/templates/itps/version_nov22/martini_v3.0_diacerolglycolipids_dev_v${version}.itp .
    cp /martini/rubiz/thylakoid/templates/itps/version_nov22/martini_v3.0_phospholipids_PG_dev_v${version}.itp .
    copy_restraints "head"
    write_thylakoid_itps "system.top"
    write_glycolipids_run 
}

prepare_run_stacks(){
    version=alpha
    cp /martini/rubiz/thylakoid/templates/itps/version_nov22/martini_v3.0_diacerolglycolipids_dev_v${version}.itp .
    cp /martini/rubiz/thylakoid/templates/itps/version_nov22/martini_v3.0_phospholipids_PG_dev_v${version}.itp .
    copy_restraints "head"
    write_thylakoid_itps "system.top"
    write_stacks_run 
}




