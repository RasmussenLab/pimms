#!bin/bash
export SSHPASS=PW
sshpass -e sftp -B 258048 hela << !
   get 20191028_QX4_StEb_MA_HeLa_500ng_191029155608.raw data/
   exit
!