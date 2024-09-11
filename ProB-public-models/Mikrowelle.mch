
MACHINE Mikrowelle
SETS /* enumerated */
  TIME={sec0,sec15,sec30,min1,sec90,min2,min3,min5,min10,min15,min30,hour1,min90,hour2,hour3,hour5}
CONCRETE_CONSTANTS
  timetable,
  TN
ABSTRACT_VARIABLES
  mag,
  T,
  S,
  uhr,
  tuer,
  error,
  i
PROPERTIES
    TN : INTEGER
  & TN = 16
  & timetable : perm(TIME)
  & timetable(1) = sec0
  & timetable(2) = sec15
  & timetable(3) = sec30
  & timetable(4) = min1
  & timetable(5) = sec90
  & timetable(6) = min2
  & timetable(7) = min3
  & timetable(8) = min5
  & timetable(9) = min10
  & timetable(10) = min15
  & timetable(11) = min30
  & timetable(12) = hour1
  & timetable(13) = min90
  & timetable(14) = hour2
  & timetable(15) = hour3
  & timetable(16) = hour5
INVARIANT
    mag : BOOL
  & T : BOOL
  & S : BOOL
  & tuer : BOOL
  & error : BOOL
  & uhr : ran(timetable)
  & i : 1 .. TN
INITIALISATION
    BEGIN
         mag := FALSE
      ||
         T := FALSE
      ||
         S := FALSE
      ||
         tuer := TRUE
      ||
         error := FALSE
      ||
         i := 1
      ||
         uhr := sec0
    END
OPERATIONS
  S_drueck = 
    PRE 
        error = FALSE
    THEN
      IF S = FALSE THEN
        S := TRUE
      ELSE
        S := FALSE
      END
    END;
  
  T_drueck = 
    PRE 
        error = FALSE
      & T = FALSE
    THEN
      T := TRUE
    END;
  
  Z_drueck = 
    PRE 
        error = FALSE
      & mag = FALSE
    THEN
      IF i < TN THEN
           i := i + 1
        ||
           uhr := timetable(i + 1)
      ELSE
           i := 1
        ||
           uhr := timetable(1)
      END
    END;
  
  close = 
    PRE 
        error = FALSE
      & tuer = FALSE
    THEN
         tuer := TRUE
      ||
         T := FALSE
    END;
  
  open = 
    PRE 
        error = FALSE
      & mag = FALSE
      & tuer = TRUE
      & T = TRUE
    THEN
      tuer := FALSE
    END;
  
  start = 
    PRE 
        error = FALSE
      & S = TRUE
      & mag = FALSE
      & tuer = TRUE
      & i >= 2
    THEN
         mag := TRUE
      ||
         S := FALSE
    END;
  
  laufe = 
    PRE 
        error = FALSE
      & mag = TRUE
      & S = FALSE
      & i >= 1
    THEN
      IF i > 2 THEN
           i := i - 1
        ||
           uhr := timetable(i - 1)
      ELSE
           mag := FALSE
        ||
           i := 1
        ||
           uhr := timetable(1)
      END
    END;
  
  stop = 
    PRE 
        error = FALSE
      & mag = TRUE
      & S = TRUE
    THEN
         mag := FALSE
      ||
         S := FALSE
    END;
  
  fehler = 
    PRE 
        error = FALSE
    THEN
         error := TRUE
      ||
         mag := FALSE
    END;
  
  reset = 
    PRE 
        error = TRUE
    THEN
      error,mag,i,uhr := FALSE,FALSE,1,timetable(1)
    END;
  
  reset_all = 
    BEGIN
         mag := FALSE
      ||
         T := FALSE
      ||
         S := FALSE
      ||
         tuer := TRUE
      ||
         error := FALSE
      ||
         i := 1
      ||
         uhr := sec0
    END;
  
  Zeit,Status <-- anzeige = 
    BEGIN
         Zeit := uhr
      ||
         IF mag = TRUE THEN
           Status := "Ein"
         ELSE
           Status := "Aus"
         END
    END
END
