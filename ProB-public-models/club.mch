
MACHINE club
SETS /* enumerated */
  NAMES={NAMES1,NAMES2,NAMES3,NAMES4,NAMES5,NAMES6}
CONCRETE_CONSTANTS
  bound
ABSTRACT_VARIABLES
  member,
  waiting
PROPERTIES
    bound : INTEGER
  & bound = 500
INVARIANT
    member : POW(NAMES)
  & waiting : POW(NAMES)
  & card(member) <= bound
  & member /\ waiting = {}
INITIALISATION
    BEGIN
         member := {}
      ||
         waiting := {}
    END
OPERATIONS
  join(nn) = 
    PRE 
        nn : waiting
      & card(member) < bound
    THEN
      member,waiting := member \/ {nn},waiting - {nn}
    END;
  
  join_queue(nn) = 
    PRE 
        nn : NAMES
      & nn /: waiting
      & nn /: member
    THEN
      waiting := waiting \/ {nn}
    END;
  
  remove(nn) = 
    PRE 
        nn : member
    THEN
      member := member - {nn}
    END;
  
  remove_queue(nn) = 
    PRE 
        nn : waiting
    THEN
      waiting := waiting - {nn}
    END;
  
  jump_queue(nn) = 
    PRE 
        nn : NAMES
      & nn /: waiting
      & nn /: member
      & card(member) < bound
    THEN
      member := member \/ {nn}
    END;
  
  nn <-- query_membership = 
    BEGIN
      nn := card(member)
    END;
  
  nn <-- lpquery_queue = 
    BEGIN
      nn := card(waiting)
    END
END
