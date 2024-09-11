
MACHINE monitor2
SETS /* enumerated */
  Staff={john,peter,mary,michael,s01,s02}
ABSTRACT_VARIABLES
  in,
  out,
  users
INVARIANT
    in : POW(Staff)
  & out : POW(Staff)
  & users : POW(Staff)
  & (users = in\/out & in/\out = {})
  & card(in) <= 4
INITIALISATION
    BEGIN
      in,out,users := {},{},{}
    END
OPERATIONS
  enter(ss) = 
    PRE 
        ss : out
      & card(in) < 4
    THEN
      in,out := in \/ {ss},out - {ss}
    END;
  
  leave(ss) = 
    PRE 
        ss : in
    THEN
      in,out := in - {ss},out \/ {ss}
    END;
  
  ss1 <-- newuser = 
    ANY ss
    WHERE 
        ss : Staff
      & ss /: users
    THEN
      users,out,ss1 := users \/ {ss},out \/ {ss},ss
    END;
  
  removeuser(ss) = 
    PRE 
        ss : users
    THEN
         users := users - {ss}
      ||
         IF ss : in THEN
           in := in - {ss}
         ELSE
           out := out - {ss}
         END
    END
END
