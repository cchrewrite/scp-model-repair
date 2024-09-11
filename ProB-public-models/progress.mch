
MACHINE progress
SETS /* enumerated */
  NAME={NM1,NM2,NM3,NM4,NM5}
ABSTRACT_VARIABLES
  trainee,
  sales,
  manager
INVARIANT
    trainee : POW(NAME)
  & sales : POW(NAME)
  & manager : POW(NAME)
  & trainee /\ sales = {}
  & trainee /\ manager = {}
  & sales /\ manager = {}
INITIALISATION
    BEGIN
      trainee,sales,manager := {},{},{}
    END
OPERATIONS
  join(ee) = 
    PRE 
        ee : NAME
      & ee /: (trainee \/ sales \/ manager)
    THEN
      trainee := trainee \/ {ee}
    END;
  
  promotetrainee(ee) = 
    PRE 
        ee : trainee
    THEN
      trainee,sales := trainee - {ee},sales \/ {ee}
    END;
  
  promotesales(ee) = 
    PRE 
        ee : sales
    THEN
      sales,manager := sales - {ee},manager \/ {ee}
    END;
  
  fired(ee) = 
    PRE 
        ee : trainee \/ sales \/ manager
    THEN
      IF ee : trainee THEN
        trainee := trainee - {ee}
      ELSIF ee : sales THEN
        sales := sales - {ee}
      ELSE
        manager := manager - {ee}
      END
    END
END
