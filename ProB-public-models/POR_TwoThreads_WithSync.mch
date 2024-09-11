
MACHINE POR_TwoThreads_WithSync
CONCRETE_CONSTANTS
  n
ABSTRACT_VARIABLES
  pc1,
  pc2,
  v1,
  v2
PROPERTIES
    n : INTEGER
  & n = 50
INVARIANT
    v1 : INTEGER
  & v2 : INTEGER
  & pc1 : NATURAL
  & pc2 : NATURAL
INITIALISATION
    BEGIN
      pc1,pc2,v1,v2 := 0,0,0,0
    END
OPERATIONS
  Step1_p1 = 
    SELECT 
        pc1 < n
    THEN 
         pc1 := pc1 + 1
      ||
         v1 := v1 + 1
    END;
  
  Step1_p2 = 
    SELECT 
        pc2 < n
    THEN 
         pc2 := pc2 + 1
      ||
         v2 := v2 + 1
    END;
  
  Sync = 
    SELECT 
        pc1 = n
      & pc2 = n
    THEN 
      pc1,pc2,v1,v2 := 0,0,v1 mod 2,v2 mod 2
    END
END
