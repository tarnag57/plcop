fof(t48_yellow_0, conjecture,  (! [A] :  ( ( ~ (v2_struct_0(A))  & l1_orders_2(A))  =>  (! [B] :  (! [C] :  ( ( (! [D] :  (m1_subset_1(D, u1_struct_0(A)) =>  (r1_lattice3(A, B, D) <=> r1_lattice3(A, C, D)) ) )  & r2_yellow_0(A, B))  => r2_yellow_0(A, C)) ) ) ) ) ).
fof(d8_yellow_0, axiom,  (! [A] :  (l1_orders_2(A) =>  (! [B] :  (r2_yellow_0(A, B) <=>  (? [C] :  (m1_subset_1(C, u1_struct_0(A)) &  (r1_lattice3(A, B, C) &  ( (! [D] :  (m1_subset_1(D, u1_struct_0(A)) =>  (r1_lattice3(A, B, D) => r1_orders_2(A, D, C)) ) )  &  (! [D] :  (m1_subset_1(D, u1_struct_0(A)) =>  ( (r1_lattice3(A, B, D) &  (! [E] :  (m1_subset_1(E, u1_struct_0(A)) =>  (r1_lattice3(A, B, E) => r1_orders_2(A, E, D)) ) ) )  => D=C) ) ) ) ) ) ) ) ) ) ) ).