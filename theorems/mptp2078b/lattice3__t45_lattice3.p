% Mizar problem: t45_lattice3,lattice3,1818,46 
fof(t45_lattice3, conjecture,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v10_lattices(A) &  (v4_lattice3(A) & l3_lattices(A)) ) )  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(A)) =>  (B=k15_lattice3(A, a_2_3_lattice3(A, B)) & B=k16_lattice3(A, a_2_4_lattice3(A, B))) ) ) ) ) ).
fof(antisymmetry_r2_hidden, axiom,  (! [A, B] :  (r2_hidden(A, B) =>  ~ (r2_hidden(B, A)) ) ) ).
fof(d16_lattice3, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  & l3_lattices(A))  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(A)) =>  (! [C] :  (r3_lattice3(A, B, C) <=>  (! [D] :  (m1_subset_1(D, u1_struct_0(A)) =>  (r2_hidden(D, C) => r1_lattices(A, B, D)) ) ) ) ) ) ) ) ) ).
fof(d17_lattice3, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  & l3_lattices(A))  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(A)) =>  (! [C] :  (r4_lattice3(A, B, C) <=>  (! [D] :  (m1_subset_1(D, u1_struct_0(A)) =>  (r2_hidden(D, C) => r1_lattices(A, D, B)) ) ) ) ) ) ) ) ) ).
fof(dt_k15_lattice3, axiom,  (! [A, B] :  ( ( ~ (v2_struct_0(A))  & l3_lattices(A))  => m1_subset_1(k15_lattice3(A, B), u1_struct_0(A))) ) ).
fof(dt_k16_lattice3, axiom,  (! [A, B] :  ( ( ~ (v2_struct_0(A))  & l3_lattices(A))  => m1_subset_1(k16_lattice3(A, B), u1_struct_0(A))) ) ).
fof(dt_k1_xboole_0, axiom, $true).
fof(dt_l1_lattices, axiom,  (! [A] :  (l1_lattices(A) => l1_struct_0(A)) ) ).
fof(dt_l1_struct_0, axiom, $true).
fof(dt_l2_lattices, axiom,  (! [A] :  (l2_lattices(A) => l1_struct_0(A)) ) ).
fof(dt_l3_lattices, axiom,  (! [A] :  (l3_lattices(A) =>  (l1_lattices(A) & l2_lattices(A)) ) ) ).
fof(dt_m1_subset_1, axiom, $true).
fof(dt_o_0_0_xboole_0, axiom, v1_xboole_0(o_0_0_xboole_0)).
fof(dt_u1_struct_0, axiom, $true).
fof(existence_l1_lattices, axiom,  (? [A] : l1_lattices(A)) ).
fof(existence_l1_struct_0, axiom,  (? [A] : l1_struct_0(A)) ).
fof(existence_l2_lattices, axiom,  (? [A] : l2_lattices(A)) ).
fof(existence_l3_lattices, axiom,  (? [A] : l3_lattices(A)) ).
fof(existence_m1_subset_1, axiom,  (! [A] :  (? [B] : m1_subset_1(B, A)) ) ).
fof(fraenkel_a_2_1_lattice3, axiom,  (! [A, B, C] :  ( ( ~ (v2_struct_0(B))  & l3_lattices(B))  =>  (r2_hidden(A, a_2_1_lattice3(B, C)) <=>  (? [D] :  (m1_subset_1(D, u1_struct_0(B)) &  (A=D & r3_lattice3(B, D, C)) ) ) ) ) ) ).
fof(fraenkel_a_2_3_lattice3, axiom,  (! [A, B, C] :  ( ( ( ~ (v2_struct_0(B))  &  (v10_lattices(B) &  (v4_lattice3(B) & l3_lattices(B)) ) )  & m1_subset_1(C, u1_struct_0(B)))  =>  (r2_hidden(A, a_2_3_lattice3(B, C)) <=>  (? [D] :  (m1_subset_1(D, u1_struct_0(B)) &  (A=D & r3_lattices(B, D, C)) ) ) ) ) ) ).
fof(fraenkel_a_2_4_lattice3, axiom,  (! [A, B, C] :  ( ( ( ~ (v2_struct_0(B))  &  (v10_lattices(B) &  (v4_lattice3(B) & l3_lattices(B)) ) )  & m1_subset_1(C, u1_struct_0(B)))  =>  (r2_hidden(A, a_2_4_lattice3(B, C)) <=>  (? [D] :  (m1_subset_1(D, u1_struct_0(B)) &  (A=D & r3_lattices(B, C, D)) ) ) ) ) ) ).
fof(redefinition_r3_lattices, axiom,  (! [A, B, C] :  ( ( ( ~ (v2_struct_0(A))  &  (v6_lattices(A) &  (v8_lattices(A) &  (v9_lattices(A) & l3_lattices(A)) ) ) )  &  (m1_subset_1(B, u1_struct_0(A)) & m1_subset_1(C, u1_struct_0(A))) )  =>  (r3_lattices(A, B, C) <=> r1_lattices(A, B, C)) ) ) ).
fof(reflexivity_r3_lattices, axiom,  (! [A, B, C] :  ( ( ( ~ (v2_struct_0(A))  &  (v6_lattices(A) &  (v8_lattices(A) &  (v9_lattices(A) & l3_lattices(A)) ) ) )  &  (m1_subset_1(B, u1_struct_0(A)) & m1_subset_1(C, u1_struct_0(A))) )  => r3_lattices(A, B, B)) ) ).
fof(t1_subset, axiom,  (! [A] :  (! [B] :  (r2_hidden(A, B) => m1_subset_1(A, B)) ) ) ).
fof(t2_subset, axiom,  (! [A] :  (! [B] :  (m1_subset_1(A, B) =>  (v1_xboole_0(B) | r2_hidden(A, B)) ) ) ) ).
fof(t2_tarski, axiom,  (! [A] :  (! [B] :  ( (! [C] :  (r2_hidden(C, A) <=> r2_hidden(C, B)) )  => A=B) ) ) ).
fof(t41_lattice3, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v10_lattices(A) &  (v4_lattice3(A) & l3_lattices(A)) ) )  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(A)) =>  (! [C] :  ( (r2_hidden(B, C) & r4_lattice3(A, B, C))  => k15_lattice3(A, C)=B) ) ) ) ) ) ).
fof(t42_lattice3, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v10_lattices(A) &  (v4_lattice3(A) & l3_lattices(A)) ) )  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(A)) =>  (! [C] :  ( (r2_hidden(B, C) & r3_lattice3(A, B, C))  => k16_lattice3(A, C)=B) ) ) ) ) ) ).
fof(t6_boole, axiom,  (! [A] :  (v1_xboole_0(A) => A=k1_xboole_0) ) ).
fof(t7_boole, axiom,  (! [A] :  (! [B] :  ~ ( (r2_hidden(A, B) & v1_xboole_0(B)) ) ) ) ).
fof(t8_boole, axiom,  (! [A] :  (! [B] :  ~ ( (v1_xboole_0(A) &  ( ~ (A=B)  & v1_xboole_0(B)) ) ) ) ) ).
fof(cc1_lattices, axiom,  (! [A] :  (l3_lattices(A) =>  ( ( ~ (v2_struct_0(A))  & v10_lattices(A))  =>  ( ~ (v2_struct_0(A))  &  (v4_lattices(A) &  (v5_lattices(A) &  (v6_lattices(A) &  (v7_lattices(A) &  (v8_lattices(A) & v9_lattices(A)) ) ) ) ) ) ) ) ) ).