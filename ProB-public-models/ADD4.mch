
MACHINE ADD4
CONCRETE_CONSTANTS
  MaxN
ABSTRACT_VARIABLES
  inpA,
  inpB,
  inpc,
  out,
  outc,
  valid
PROPERTIES
    MaxN : INTEGER
  & MaxN = 3
INVARIANT
    valid : BOOL
  & inpA : 0 .. MaxN
  & inpB : 0 .. MaxN
  & inpc : 0 .. 1
  & out : 0 .. MaxN
  & outc : 0 .. 1
  & (valid = TRUE => outc * (MaxN + 1) + out = inpA + inpB + inpc)
INITIALISATION
    BEGIN
         inpA := 0
      ||
         inpB := 0
      ||
         inpc := 0
      ||
         out := 0
      ||
         outc := 0
      ||
         valid := TRUE
    END
OPERATIONS
  calculate = 
    SELECT 
        valid = FALSE
    THEN 
      out,outc,valid := (inpA + inpB + inpc) mod (MaxN + 1),(inpA + inpB + inpc) / (MaxN + 1),TRUE
    END;
  
  reset = 
    SELECT 
        valid = TRUE
    THEN 
         valid := FALSE
      ||
         ANY ia,ib,ic
         WHERE 
             ia : 0 .. MaxN
           & ib : 0 .. MaxN
           & ic : 0 .. 1
         THEN
              inpA := ia
           ||
              inpB := ib
           ||
              inpc := ic
         END
    END;
  
  evol0 = 
    skip;
  
  setc1 = 
    skip;
  
  evol1 = 
    skip;
  
  setc2 = 
    skip;
  
  evol2 = 
    skip;
  
  setc3 = 
    skip;
  
  evol3 = 
    skip
END
