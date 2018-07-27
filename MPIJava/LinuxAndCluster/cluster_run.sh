mark=test1 

# path
output_path=/bigstore/hlcm2/tianzhiliang/latentTree/output
working_path=/bigstore/hlcm2/tianzhiliang/latentTree/cluster_running_dir/
working_path_str="\/bigstore\/hlcm2\/tianzhiliang\/latentTree\/cluster_running_dir\/"
bin_dir=bin

model_file_prefix=${output_path}/model_${mark}
working_dir=${working_path}/${mark}
working_dir_str="\#\$ -wd "${working_path_str}"\/"${mark}
output_dir_str="\#\$ -o "${working_path_str}"\/"${mark}"\/output_dir"

mkdir -p ${working_dir}
cp -r ${bin_dir} ${working_dir}

job_id_for_shell_name=${mark}
cp local_called_by_cluster.sh ${job_id_for_shell_name}

cp ${job_id_for_shell_name} ${job_id_for_shell_name}.sed
sed -i "s/this_is_output_path/$output_dir_str/" ${job_id_for_shell_name}.sed
sed -i "s/this_is_working_path/$working_dir_str/" ${job_id_for_shell_name}.sed
mv ${job_id_for_shell_name}.sed  ${job_id_for_shell_name}
chmod 777 ${job_id_for_shell_name}

qsub ${job_id_for_shell_name} ${data} ${model_file_prefix} ${logout} ${logerr}
rm ${job_id_for_shell_name}
