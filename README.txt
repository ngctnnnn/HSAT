HUONG DAN SU DUNG - Table 1 FULL run (15 to hop dataset x model, x4 method)
====================================================================

DA XAC NHAN VOI AUTHOR (khong con GUESSED nua)
====================================================================
- warmup = 20 cho toan bo 9 to hop truoc day chua co config (PEMS03/FEDformer,
  PEMS04/TimesNet, Weather/Autoformer, PEMS08 x3 model, ETTm1 x3 model)
- hidden_dim, alpha_t, epsilon: author xac nhan "set giong nhu may model khac" 
  => dung gia tri pho bien nhat trong cac file da co (hidden_dim=64, 
  alpha_t=0.03, epsilon=0.2)

Toan bo 15 file config gio deu duoc coi la CONFIRMED, khong con la doan mu.


VAN DE QUAN TRONG THU HAI - DOC TRUOC KHI CHAY (checkpoint Clean/Backtime)
====================================================================
Truoc khi chay bat ky lenh nao, PHAI xac nhan run.py tren server DA duoc
fix loi trung ten checkpoint giua Clean va Backtime:

  Kiem tra: grep -n "save_file = " run.py
  Phai thay dong nay co "_clean={no_attack}" o gan cuoi, truoc ".pth"

Neu CHUA fix: 15 cap Clean/Backtime moi trong bo lenh nay se bi dinh loi
Clean va Backtime dung chung 1 checkpoint, ra ket qua giong het nhau.

Neu DA fix nhung con checkpoint CU (tao truoc khi fix) con sot lai duoi
ten khong co "_clean=...", can xoa het truoc khi chay:
  rm -f checkpoints/*TgrGCN*.pth
(Backtime dung type=TgrGCN, day la loai duy nhat bi trung voi Clean.
MSAT/HSAT dung TgrCNN1D/TgrGLGCN, khong bi anh huong, KHONG can xoa.)


1. CACH DUNG
====================================================================
- Copy 15 file trong configs/backdoor_config/full_table1_*.yaml vao
  ~/maikhanh/HSATproject-main/configs/backdoor_config/
- Copy 3 file trong scripts/ vao ~/maikhanh/HSATproject-main/scripts/
- Dam bao da co configs/surrogate_config/clean_config.yaml (voi
  save_file fix da ap dung trong run.py)
- Chay tung file trong tmux:
    bash scripts/Table1_FULL_Autoformer_submit_job.sh
    bash scripts/Table1_FULL_FEDformer_submit_job.sh
    bash scripts/Table1_FULL_TimesNet_submit_job.sh
  Moi file: 5 dataset x 4 method = 20 lenh, ~1-1.5h/lenh => 20-30h/file


2. CHUA BAO GOM TRONG BO NAY
====================================================================
- TDBA (baseline): chay rieng qua TDBA/main.py, khong dung run.py
- Chay lap 3 seed: sau khi co ket qua seed hien tai, sua field "seed:"
  trong 4 file surrogate_config (default_config.yaml,
  surrogate_config_TgrCNN1D.yaml, surrogate_config_TgrGLGCN.yaml,
  clean_config.yaml) roi chay lai toan bo, giong cach da lam truoc do.
