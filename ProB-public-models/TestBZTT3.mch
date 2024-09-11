
MACHINE TestBZTT3
SETS /* enumerated */
  ID={a,b,c,d,e,f,g,h,i,j}
ABSTRACT_VARIABLES
  fun,
  x,
  s
INVARIANT
    x : ID
  & s : POW(ID)
  & fun : ID --> ID
INITIALISATION
    BEGIN
         fun := {a |-> b,b |-> c,c |-> d,d |-> e,e |-> f,f |-> g,g |-> h,h |-> i,i |-> j,j |-> a}
      ||
         x := a
      ||
         s := ID
    END
OPERATIONS
  ApplyF = 
    BEGIN
      x := fun(x)
    END;
  
  Fixp(z) = 
    PRE 
        z : ID
      & z = fun(z)
    THEN
      x := z
    END;
  
  Find(T) = 
    PRE 
        T : ID
      & fun(T) = x
    THEN
      x := T
    END;
  
  Iterate = 
    BEGIN
      fun := (fun ; fun)
    END;
  
  Solve1(ff) = 
    PRE 
        ff : ID --> ID
      & ff(a) /= ff(b)
      & ff(b) /= ff(c)
      & ff(c) /= ff(d)
      & ff(d) /= ff(e)
      & ff(e) /= ff(g)
      & ff(g) = a
      & ran(ff) = {a,b}
    THEN
      fun := ff
    END;
  
  UnSolveable1(ff) = 
    PRE 
        ff : ID --> ID
      & ff(a) /= ff(b)
      & ff(b) /= ff(c)
      & ff(c) /= ff(d)
      & ff(d) /= ff(e)
      & ff(e) /= ff(g)
      & ff(g) = a
      & ff(a) = a
      & ran(ff) = {a,b}
    THEN
      fun := ff
    END;
  
  Solve2(ff) = 
    PRE 
        ff : ID --> ID
      & (ff ; ff) = ff
      & ran(ff) = {a}
    THEN
      fun := ff
    END
END
