
MACHINE CSM
ABSTRACT_VARIABLES
  x0,
  x1,
  x2,
  x3,
  x4,
  x5,
  x6,
  x7,
  x8,
  x9,
  x10,
  x11,
  x12,
  x13
INVARIANT
    x0 : NATURAL
  & x1 : NATURAL
  & x2 : NATURAL
  & x3 : NATURAL
  & x4 : NATURAL
  & x5 : NATURAL
  & x6 : NATURAL
  & x7 : NATURAL
  & x8 : NATURAL
  & x9 : NATURAL
  & x10 : NATURAL
  & x11 : NATURAL
  & x12 : NATURAL
  & x13 : NATURAL
  & x9 < 2
INITIALISATION
    BEGIN
         x0 := 0
      ||
         x1 := 0
      ||
         x2 := 0
      ||
         x3 := 0
      ||
         x4 := 0
      ||
         x5 := 1
      ||
         x6 := 1
      ||
         x7 := 4
      ||
         x8 := 0
      ||
         x9 := 0
      ||
         x10 := 0
      ||
         x11 := 0
      ||
         x12 := 0
      ||
         x13 := 1
    END
OPERATIONS
  t0 = 
    PRE 
        x0 >= 1
    THEN
         x0 := x0 - 1
      ||
         x1 := x1 + 1
    END;
  
  t1 = 
    PRE 
        x4 >= 1
      & x9 >= 1
    THEN
         x0 := x0 + 1
      ||
         x2 := x2 + 1
      ||
         x4 := x4 - 1
      ||
         x9 := x9 - 1
    END;
  
  t2 = 
    PRE 
        x4 >= 1
      & x6 >= 1
    THEN
         x0 := x0 + 1
      ||
         x3 := x3 + 1
      ||
         x4 := x4 - 1
      ||
         x6 := x6 - 1
    END;
  
  t3 = 
    PRE 
        x1 >= 1
      & x3 >= 1
    THEN
         x1 := x1 - 1
      ||
         x3 := x3 - 1
      ||
         x5 := x5 + 1
      ||
         x6 := x6 + 1
    END;
  
  t4 = 
    PRE 
        x1 >= 1
      & x2 >= 1
    THEN
         x1 := x1 - 1
      ||
         x2 := x2 - 1
      ||
         x5 := x5 + 1
      ||
         x9 := x9 + 1
    END;
  
  t5 = 
    PRE 
        x5 >= 1
    THEN
         x4 := x4 + 1
      ||
         x5 := x5 - 1
    END;
  
  t6 = 
    PRE 
        x10 >= 1
    THEN
         x7 := x7 + 1
      ||
         x10 := x10 - 1
    END;
  
  t7 = 
    PRE 
        x7 >= 1
    THEN
         x7 := x7 - 1
      ||
         x8 := x8 + 1
    END;
  
  t8 = 
    PRE 
        x6 >= 1
      & x8 >= 1
    THEN
         x6 := x6 - 1
      ||
         x8 := x8 - 1
      ||
         x9 := x9 + 1
    END;
  
  t9 = 
    PRE 
        x9 >= 1
    THEN
         x6 := x6 + 1
      ||
         x9 := x9 - 1
      ||
         x10 := x10 + 1
    END;
  
  t10 = 
    PRE 
        x10 >= 1
    THEN
         x10 := x10 - 1
      ||
         x12 := x12 + 1
    END;
  
  t11 = 
    PRE 
        x11 >= 1
    THEN
         x8 := x8 + 1
      ||
         x11 := x11 - 1
      ||
         x13 := x13 + 1
    END;
  
  t12 = 
    PRE 
        x12 >= 1
      & x13 >= 1
    THEN
         x11 := x11 + 1
      ||
         x12 := x12 - 1
      ||
         x13 := x13 - 1
    END
END
