fof(t32_orders_2, conjecture,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v3_orders_2(A) &  (v4_orders_2(A) &  (v5_orders_2(A) & l1_orders_2(A)) ) ) )  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(A)) =>  (! [C] :  (m1_subset_1(C, u1_struct_0(A)) =>  (! [D] :  (m1_subset_1(D, k1_zfmisc_1(u1_struct_0(A))) =>  (! [E] :  (m1_subset_1(E, k1_zfmisc_1(u1_struct_0(A))) =>  ( (r2_orders_2(A, B, C) &  (r2_hidden(B, D) &  (r2_hidden(C, E) & m1_orders_2(E, A, D)) ) )  => r2_hidden(B, E)) ) ) ) ) ) ) ) ) ) ) ).
fof(d11_orders_2, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v3_orders_2(A) &  (v4_orders_2(A) &  (v5_orders_2(A) & l1_orders_2(A)) ) ) )  =>  (! [B] :  (m1_subset_1(B, k1_zfmisc_1(u1_struct_0(A))) =>  (! [C] :  (m1_subset_1(C, k1_zfmisc_1(u1_struct_0(A))) =>  ( ( ~ (B=k1_xboole_0)  =>  (m1_orders_2(C, A, B) <=>  (? [D] :  (m1_subset_1(D, u1_struct_0(A)) &  (r2_hidden(D, B) & C=k3_orders_2(A, B, D)) ) ) ) )  &  (B=k1_xboole_0 =>  (m1_orders_2(C, A, B) <=> C=k1_xboole_0) ) ) ) ) ) ) ) ) ).
fof(t24_orders_2, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v3_orders_2(A) &  (v4_orders_2(A) &  (v5_orders_2(A) & l1_orders_2(A)) ) ) )  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(A)) =>  (! [C] :  (m1_subset_1(C, u1_struct_0(A)) =>  (! [D] :  (m1_subset_1(D, k1_zfmisc_1(u1_struct_0(A))) =>  (r2_hidden(B, k3_orders_2(A, D, C)) <=>  (r2_orders_2(A, B, C) & r2_hidden(B, D)) ) ) ) ) ) ) ) ) ) ).
fof(t4_subset, axiom,  (! [A] :  (! [B] :  (! [C] :  ( (r2_hidden(A, B) & m1_subset_1(B, k1_zfmisc_1(C)))  => m1_subset_1(A, C)) ) ) ) ).
fof(t5_orders_2, axiom,  (! [A] :  ( (v4_orders_2(A) &  (v5_orders_2(A) & l1_orders_2(A)) )  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(A)) =>  (! [C] :  (m1_subset_1(C, u1_struct_0(A)) =>  (! [D] :  (m1_subset_1(D, u1_struct_0(A)) =>  ( (r2_orders_2(A, B, C) & r2_orders_2(A, C, D))  => r2_orders_2(A, B, D)) ) ) ) ) ) ) ) ) ).