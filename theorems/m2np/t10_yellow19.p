fof(t10_yellow19, conjecture,  (! [A] :  ( ( ~ (v2_struct_0(A))  & l1_struct_0(A))  =>  (! [B] :  ( ( ~ (v2_struct_0(B))  & l1_waybel_0(B, A))  =>  (! [C] :  (r2_hidden(C, k2_yellow19(A, B)) <=>  (r1_waybel_0(A, B, C) & m1_subset_1(C, k1_zfmisc_1(u1_struct_0(A)))) ) ) ) ) ) ) ).
fof(d3_yellow19, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  & l1_struct_0(A))  =>  (! [B] :  ( ( ~ (v2_struct_0(B))  & l1_waybel_0(B, A))  => k2_yellow19(A, B)=a_2_1_yellow19(A, B)) ) ) ) ).
fof(fraenkel_a_2_1_yellow19, axiom,  (! [A, B, C] :  ( ( ( ~ (v2_struct_0(B))  & l1_struct_0(B))  &  ( ~ (v2_struct_0(C))  & l1_waybel_0(C, B)) )  =>  (r2_hidden(A, a_2_1_yellow19(B, C)) <=>  (? [D] :  (m1_subset_1(D, k1_zfmisc_1(u1_struct_0(B))) &  (A=D & r1_waybel_0(B, C, D)) ) ) ) ) ) ).