
MACHINE BinomialCoefficientConcurrent
CONCRETE_CONSTANTS
  n,
  k
ABSTRACT_VARIABLES
  y1,
  y2,
  y3,
  addr1,
  addr2
PROPERTIES
    n : INTEGER
  & k : INTEGER
  & n = 100
  & k = 20
INVARIANT
    y1 : INTEGER
  & y2 : INTEGER
  & y3 : INTEGER
  & addr1 : 0 .. 3
  & addr2 : 0 .. 4
INITIALISATION
    BEGIN
         y1 := n
      ||
         y2 := 0
      ||
         y3 := 1
      ||
         addr1 := 1
      ||
         addr2 := 1
    END
OPERATIONS
  halt_1 = 
    PRE 
        addr1 = 1
    THEN
      IF y1 = n - k THEN
        addr1 := 0
      ELSE
        addr1 := 2
      END
    END;
  
  mult_1 = 
    PRE 
        addr1 = 2
    THEN
         y3 := y3 * y1
      ||
         addr1 := 3
    END;
  
  dec_1 = 
    PRE 
        addr1 = 3
    THEN
         y1 := y1 - 1
      ||
         addr1 := 1
    END;
  
  halt_2 = 
    PRE 
        addr2 = 1
    THEN
      IF y2 = k THEN
        addr2 := 0
      ELSE
        addr2 := 2
      END
    END;
  
  inc_2 = 
    PRE 
        addr2 = 2
    THEN
         y2 := y2 + 1
      ||
         addr2 := 3
    END;
  
  wait_divide_2 = 
    PRE 
        addr2 = 3
      & y2 <= n - y1
    THEN
      addr2 := 4
    END;
  
  div_2 = 
    PRE 
        addr2 = 4
    THEN
         y3 := y3 / y2
      ||
         addr2 := 1
    END;
  
  finished = 
    PRE 
        addr1 = 0
      & addr2 = 0
    THEN
      skip
    END
END
