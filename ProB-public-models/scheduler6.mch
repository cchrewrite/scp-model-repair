
MACHINE scheduler6
SETS /* enumerated */
  PID={PID1,PID2,PID3,PID4,PID5,PID6}
ABSTRACT_VARIABLES
  active,
  ready,
  waiting
INVARIANT
    active : POW(PID)
  & ready : POW(PID)
  & waiting : POW(PID)
  & ready /\ waiting = {}
  & active /\ (ready \/ waiting) = {}
  & card(active) <= 1
  & (active = {} => ready = {})
INITIALISATION
    BEGIN
         active := {}
      ||
         ready := {}
      ||
         waiting := {}
    END
OPERATIONS
  rr <-- nr_ready = 
    BEGIN
      rr := card(ready)
    END;
  
  new(pp) = 
    PRE 
        pp : PID
    THEN
      SELECT 
          pp /: active
        & pp /: (ready \/ waiting)
      THEN 
        waiting := waiting \/ {pp}
      END
    END;
  
  del(pp) = 
    PRE 
        pp : PID
    THEN
      SELECT 
          pp : waiting
      THEN 
        waiting := waiting - {pp}
      END
    END;
  
  readyp(rr) = 
    PRE 
        rr : PID
    THEN
      SELECT 
          rr : waiting
      THEN 
           waiting := waiting - {rr}
        ||
           IF active = {} THEN
             active := {rr}
           ELSE
             ready := ready \/ {rr}
           END
      END
    END;
  
  swap = 
    SELECT 
        active /= {}
    THEN 
         waiting := waiting \/ active
      ||
         IF ready = {} THEN
           active := {}
         ELSE
           ANY pp
           WHERE 
               pp : ready
           THEN
                active := {pp}
             ||
                ready := ready - {pp}
           END
         END
    END
END
