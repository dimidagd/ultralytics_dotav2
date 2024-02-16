#!/bin/bash
print_ok() {
  # Define color codes
  GREEN='\033[0;32m'
  NC='\033[0m' # No Color

  # Print the green [OK] in front of the argument
  printf "${GREEN}[OK]${NC} %s\n" "$1"
}


print_failed() {
  # Define color codes
  RED='\033[0;31m'
  NC='\033[0m' # No Color


  # Print the green [OK] in front of the argument
  printf "[${RED}FAILED${NC}] %s\n" "$1"
}


not_found() {
  # Define color codes
  RED='\033[0;31m'
  NC='\033[0m' # No Color

  # Print custom message
  echo -e "${RED}$1${NC} not found, available experiments are:"

  # Print list of available experiments
  echo "----------- AVAILABLE EXPERIMENTS ----------"
  find ../notebooks/config/experiment/*.yaml -printf "-> %f\n"
  echo "--------------------------------------------"
}

choose_experiment() {
  # Define color codes
  CYAN='\033[0;36m'
  GREEN='\033[0;32m'
  NC='\033[0m' # No Color

  # Save the list of available experiments in an array
  mapfile -t experiments < <(find ../notebooks/config/experiment/*.yaml -printf "-> %f\n")

  # Initialize variables
  selected_index=0
  num_experiments=${#experiments[@]}
  scroll_offset=0
  saved_line=$LINENO
  # Function to print the experiments list with highlight
  print_experiments() {
    tput cup $((2 + scroll_offset)) 0
    for ((i=scroll_offset; i<scroll_offset+num_experiments; i++)); do
      if [[ $i -eq $((scroll_offset + selected_index)) ]]; then
        echo -ne "${GREEN}${experiments[$i - scroll_offset]}${NC}"
      else
        echo -ne "${CYAN}${experiments[$i - scroll_offset]}${NC}"
      fi
      echo -e "\033[K"  # Clear the rest of the line
      tput el  # Clear the rest of the line (alternative)
    done
  }

  # Print prompt and available experiments
  echo -e "Available experiments:"
  print_experiments

  # Process user input
  while read -r -sn1 key; do
    case "$key" in
      A)  # Up arrow
          if [[ $selected_index -gt 0 ]]; then
            selected_index=$((selected_index - 1))
            if [[ $selected_index -lt $scroll_offset ]]; then
              scroll_offset=$((scroll_offset - 1))
            fi
            print_experiments
          fi
          ;;
      B)  # Down arrow
          if [[ $selected_index -lt $((num_experiments - 1)) ]]; then
            selected_index=$((selected_index + 1))
            if [[ $selected_index -ge $((scroll_offset + num_experiments)) ]]; then
              scroll_offset=$((scroll_offset + 1))
            fi
            print_experiments
          fi
          ;;
      '') # Enter key
          clear
          experiment_file="${experiments[scroll_offset + selected_index]#-> }"
          experiment_name="${experiment_file%.yaml}"
          export EXPERIMENT=$experiment_name
          tput cup $((2 + scroll_offset + selected_index)) 0
          echo -e "\nSelected experiment: $EXPERIMENT"
          break
          ;;
    esac
  done
}

function setup_remote_environment() {
  MOUNT_MODE=${1:-rw} # Give rw as argument to allow write to the Workspace directory with ENC,XBR imgs etc
  if [ ! -f "/ssh_dir/.ssh/id_rsa_custom_sfugroup" ]; then
    cd .. && ./setup-ssh.sh && print_ok "Setup-ssh keys successful" && cd scripts || print_failed "Setting up ssh keys failed"
  fi

  echo $(date)
  echo "PWD: $PWD"
  echo "Log level: $LOG_LEVEL"
  set -a  # Automatically export all variables
  source .env
  set +a  # Stop automatically exporting variables
  cat .env
  mkdir -p /root/.ssh
  cp -r /ssh_dir/.ssh/* /root/.ssh/
  chmod 700 /root/.ssh && \
  chmod 600 /root/.ssh/* && \
  eval `ssh-agent`
  SSHPRIVATEKEY=/root/.ssh/id_rsa_custom_sfugroup

  if [ -f "$SSHPRIVATEKEY" ]; then
      print_ok "$SSHPRIVATEKEY exists, good job."
  else
      print_failed "WARNING: $SSHPRIVATEKEY does not exist, did you run ./setup-ssh.sh?"
  fi
  ssh-add /root/.ssh/id_rsa_custom_sfugroup
  print_ok "SSH keys copied from host, permissions applied" \
  || (print_failed "Copying ssh keys from host failed" && exit 1)

  id=`lsof -t -i @localhost:5432 -sTCP:listen`
  [ ! -z "$id" ] && kill -9 $id && print_ok "Killed any previous port tunneling on localhost:5432"
  ssh -4 -fNT -F /dev/null -o StrictHostKeyChecking=no -L  5432:localhost:5432 ${NUC_user}@${NUC_ip} && \
  print_ok "Remote DB tunneled locally to port 5432" \
  || (print_failed "Tunneling of port 5432 from containers host failed." && exit 1)
  mountpoint -q $local_dir && umount -l $local_dir
  mkdir -p $local_dir
  sshfs -o ssh_command="ssh -J  ${NUC_user}@${NUC_ip}",auto_cache -o Ciphers=aes128-ctr ${nas_username}@${NAS_hostname}:$remote_dir $local_dir -o StrictHostKeyChecking=no -o password_stdin -o uid=1000 -o ro <<< $nas_pass && \
  print_ok "Remote filesystem  ${nas_username}@${NAS_hostname}:$remote_dir mounted locally at $local_dir with sshfs(1/2)" \
  || (print_failed "sshfs ${nas_username}@${NAS_hostname}:$remote_dir at $local_dir failed" && exit 1)
  mountpoint -q ${DATAWAREHOUSE_LOCAL_DIR} && umount -l ${DATAWAREHOUSE_LOCAL_DIR}
  mkdir -p ${DATAWAREHOUSE_LOCAL_DIR}
  #sshfs -o auto_cache -o Ciphers=aes128-ctr ${NUC_user}@${NUC_ip}:${DATAWAREHOUSE_REMOTE_DIR} ${DATAWAREHOUSE_LOCAL_DIR} -o allow_other && \
  sshfs -o ssh_command="ssh -J  ${NUC_user}@${NUC_ip}",auto_cache -o Ciphers=aes128-ctr ${nas_username}@${NAS_hostname}:${DATAWAREHOUSE_REMOTE_DIR} ${DATAWAREHOUSE_LOCAL_DIR} -o StrictHostKeyChecking=no -o password_stdin -o uid=1000 -o $MOUNT_MODE <<< $nas_pass && \
  print_ok "Remote filesystem ${nas_username}@${NAS_hostname}:${DATAWAREHOUSE_REMOTE_DIR} mounted locally at ${DATAWAREHOUSE_LOCAL_DIR} with sshfs(2/2)" \
  || (print_failed "sshfs ${nas_username}@${NAS_hostname}:${DATAWAREHOUSE_REMOTE_DIR} mounted locally at ${DATAWAREHOUSE_LOCAL_DIR} failed" && exit 1)
}

# Example usage
#choose_experiment
