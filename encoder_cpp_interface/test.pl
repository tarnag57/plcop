% This is a test file for interfacing cpp. It has no utility.

:- load_foreign_library('encoder.so').

result(X) :- encoder:encode_clause([-v1_xxreal_0(sklm), var=var], X).