% This is a test file for interfacing cpp. It has no utility.

:- load_foreign_library('encoder.so').
result(X) :- encoder:encode_clause([r2_hidden(3^[],4^[]),k1_tarski(3^[])=k1_tarski(_39396),- (2^[4^[],k1_tarski(3^[])]=_39396)], X).
empty_res(X) :- encoder:encode_clause([], X).
failure_res(X) :- encoder:encode_clause([failure], X).