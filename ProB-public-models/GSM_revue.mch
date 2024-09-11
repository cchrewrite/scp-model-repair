
MACHINE GSM_revue
SETS /* enumerated */
  FILES={mf,df_gsm,ef_iccid,ef_lp,ef_imsi,ef_ad};
  PERMISSION={always,chv,never,adm};
  VALUE={true,false};
  BLOCKED_STATUS={blocked,unblocked};
  CODE={a_1,a_2,a_3,a_4};
  DATA={d_1,d_2,d_3,d_4,d_err}
CONCRETE_CONSTANTS
  FILES_CHILDREN,
  PERMISSION_READ,
  MAX_CHV,
  MAX_UNBLOCK,
  PUK
ABSTRACT_VARIABLES
  current_file,
  current_directory,
  counter_chv,
  counter_unblock_chv,
  blocked_chv_status,
  blocked_status,
  permission_session,
  pin,
  data
PROPERTIES
    FILES_CHILDREN : POW(FILES * FILES)
  & PUK : CODE
  & FILES_CHILDREN = {mf |-> df_gsm,mf |-> ef_iccid,df_gsm |-> ef_lp,df_gsm |-> ef_imsi,df_gsm |-> ef_ad}
  & PERMISSION_READ : {ef_iccid,ef_lp,ef_imsi,ef_ad} --> PERMISSION
  & PERMISSION_READ = {ef_iccid |-> never,ef_lp |-> always,ef_imsi |-> chv,ef_ad |-> adm}
  & MAX_CHV = 2
  & PERMISSION_READ : FILES +-> PERMISSION
  & MAX_CHV : NATURAL
  & MAX_UNBLOCK : NATURAL
  & MAX_UNBLOCK = 3
  & PUK = a_3
INVARIANT
    blocked_chv_status : BLOCKED_STATUS
  & blocked_status : BLOCKED_STATUS
  & pin : CODE
  & data : FILES +-> DATA
  & current_file <: {ef_iccid,ef_lp,ef_imsi,ef_ad}
  & card(current_file) <= 1
  & current_directory : {df_gsm} \/ {mf}
  & counter_chv : 0 .. MAX_CHV
  & counter_unblock_chv : 0 .. MAX_UNBLOCK
  & permission_session : PERMISSION --> VALUE
  & always |-> true : permission_session
  & adm |-> false : permission_session
  & never |-> false : permission_session
  & data : {ef_iccid,ef_lp,ef_imsi,ef_ad} --> DATA
  & (blocked_chv_status = blocked => chv |-> false : permission_session)
  & (counter_chv = 0) <=> (blocked_chv_status = blocked)
  & (counter_unblock_chv = 0) <=> (blocked_status = blocked)
  & (current_file = {} or dom(FILES_CHILDREN |> current_file) = {current_directory})
INITIALISATION
    BEGIN
         current_file := {}
      ||
         current_directory := mf
      ||
         counter_chv := MAX_CHV
      ||
         counter_unblock_chv := MAX_UNBLOCK
      ||
         blocked_chv_status := unblocked
      ||
         blocked_status := unblocked
      ||
         permission_session := {always |-> true,chv |-> false,adm |-> false,never |-> false}
      ||
         pin := a_1
      ||
         data := {ef_iccid |-> d_1,ef_lp |-> d_2,ef_imsi |-> d_3,ef_ad |-> d_4}
    END
OPERATIONS
  sw <-- SELECT_FILE(ff) = 
    PRE 
        ff : FILES
    THEN
      IF ff : {df_gsm} \/ {mf} THEN
        IF ff |-> current_directory : FILES_CHILDREN or current_directory |-> ff : FILES_CHILDREN or #dp.(dp : {df_gsm} \/ {mf} & (dp |-> current_directory : FILES_CHILDREN & dp |-> ff : FILES_CHILDREN)) or ff = mf THEN
             sw := 9000
          ||
             current_directory := ff
          ||
             current_file := {}
        ELSE
          sw := 9404
        END
      ELSE
        IF current_directory |-> ff : FILES_CHILDREN THEN
             sw := 9000
          ||
             current_file := {ff}
        ELSE
          sw := 9404
        END
      END
    END;
  
  sw,dd <-- READ_BINARY = 
    PRE 
        current_file <: {ef_iccid,ef_lp,ef_imsi,ef_ad}
    THEN
      IF current_file = {} THEN
           sw := 9400
        ||
           dd := d_err
      ELSE
        IF #edp.(edp : PERMISSION & (PERMISSION_READ[current_file] = {edp} & permission_session(edp) = true)) THEN
             sw := 9000
          ||
             ANY ff
             WHERE 
                 ff : current_file
             THEN
               dd := data(ff)
             END
        ELSE
             sw := 9804
          ||
             dd := d_err
        END
      END
    END;
  
  cd,cf,cc,cuc <-- STATUS = 
    BEGIN
         cd := current_directory
      ||
         cf := current_file
      ||
         cc := counter_chv
      ||
         cuc := counter_unblock_chv
    END;
  
  sw <-- VERIFY_CHV(code) = 
    PRE 
        code : CODE
    THEN
      IF blocked_chv_status = blocked THEN
        sw := 9840
      ELSE
        IF pin = code THEN
             counter_chv := MAX_CHV
          ||
             permission_session(chv) := true
          ||
             sw := 9000
        ELSE
          IF counter_chv = 1 THEN
               counter_chv := 0
            ||
               blocked_chv_status := blocked
            ||
               permission_session(chv) := false
            ||
               sw := 9840
          ELSE
               counter_chv := counter_chv - 1
            ||
               sw := 9804
          END
        END
      END
    END;
  
  sw <-- UNBLOCK_CHV(code_unblock,new_code) = 
    PRE 
        code_unblock : CODE
      & new_code : CODE
    THEN
      IF blocked_status = blocked THEN
        sw := 9840
      ELSE
        IF PUK = code_unblock THEN
             pin := new_code
          ||
             blocked_chv_status := unblocked
          ||
             counter_chv := MAX_CHV
          ||
             counter_unblock_chv := MAX_UNBLOCK
          ||
             permission_session(chv) := true
          ||
             sw := 9000
        ELSE
          IF counter_unblock_chv = 1 THEN
               counter_unblock_chv := 0
            ||
               blocked_status := blocked
            ||
               sw := 9840
          ELSE
               counter_unblock_chv := counter_unblock_chv - 1
            ||
               sw := 9804
          END
        END
      END
    END;
  
  RESET = 
    BEGIN
         current_file := {}
      ||
         current_directory := mf
      ||
         permission_session := {always |-> true,chv |-> false,adm |-> false,never |-> false}
    END
END
