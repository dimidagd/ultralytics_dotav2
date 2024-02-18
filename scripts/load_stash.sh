create_symlink() {
  local currdir=$1
  local temp_dir=$2
  mkdir -p $currdir && \
  echo "symlinking $currdir $temp_dir" && \
  ln -s $currdir $temp_dir
}


if [ $# -eq 0 ]
then
  echo "No arguments supplied, exiting"
  exit 1
fi

temp_dir=$1
set -e
echo "Untar file.."
tar -xf "$temp_dir/source_code.tar.gz" -C "$temp_dir" && echo "$temp_dir/source_code.tar.gz unstashed at $temp_dir"
rm -f $temp_dir/source_code.tar.gz
currdir=$PWD
cd $temp_dir
echo "Current working directory $PWD"

create_symlink $currdir/../examples/datasets $temp_dir/examples/
create_symlink $currdir/../runs $temp_dir/
create_symlink $currdir/../scripts/hpc_logs $temp_dir/scripts/
create_symlink $currdir/../wandb $temp_dir/

echo "Exiting, loading of stashed dir completed"
