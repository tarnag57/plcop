% Mizar problem: t33_waybel_7,waybel_7,1383,75 
fof(t33_waybel_7, conjecture,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v2_pre_topc(A) & l1_pre_topc(A)) )  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(k2_yellow_1(u1_pre_topc(A)))) =>  (! [C] :  (m1_subset_1(C, u1_struct_0(k2_yellow_1(u1_pre_topc(A)))) =>  (r1_waybel_3(k2_yellow_1(u1_pre_topc(A)), B, C) =>  (! [D] :  ( ( ~ (v1_xboole_0(D))  &  (v2_waybel_0(D, k3_yellow_1(u1_struct_0(A))) &  (v13_waybel_0(D, k3_yellow_1(u1_struct_0(A))) &  (v3_waybel_7(D, k3_yellow_1(u1_struct_0(A))) & m1_subset_1(D, k1_zfmisc_1(u1_struct_0(k3_yellow_1(u1_struct_0(A)))))) ) ) )  =>  ~ ( (r2_hidden(B, D) &  (! [E] :  (m1_subset_1(E, u1_struct_0(A)) =>  ~ ( (r2_hidden(E, C) & r2_waybel_7(A, D, E)) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ).
fof(abstractness_v1_orders_2, axiom,  (! [A] :  (l1_orders_2(A) =>  (v1_orders_2(A) => A=g1_orders_2(u1_struct_0(A), u1_orders_2(A))) ) ) ).
fof(antisymmetry_r2_hidden, axiom,  (! [A, B] :  (r2_hidden(A, B) =>  ~ (r2_hidden(B, A)) ) ) ).
fof(d2_xboole_0, axiom, k1_xboole_0=o_0_0_xboole_0).
fof(dt_g1_orders_2, axiom,  (! [A, B] :  (m1_subset_1(B, k1_zfmisc_1(k2_zfmisc_1(A, A))) =>  (v1_orders_2(g1_orders_2(A, B)) & l1_orders_2(g1_orders_2(A, B))) ) ) ).
fof(dt_k1_xboole_0, axiom, $true).
fof(dt_k1_zfmisc_1, axiom, $true).
fof(dt_k2_yellow_1, axiom,  (! [A] :  (v1_orders_2(k2_yellow_1(A)) & l1_orders_2(k2_yellow_1(A))) ) ).
fof(dt_k2_zfmisc_1, axiom, $true).
fof(dt_k3_yellow_1, axiom,  (! [A] :  (v1_orders_2(k3_yellow_1(A)) & l1_orders_2(k3_yellow_1(A))) ) ).
fof(dt_l1_orders_2, axiom,  (! [A] :  (l1_orders_2(A) => l1_struct_0(A)) ) ).
fof(dt_l1_pre_topc, axiom,  (! [A] :  (l1_pre_topc(A) => l1_struct_0(A)) ) ).
fof(dt_l1_struct_0, axiom, $true).
fof(dt_m1_subset_1, axiom, $true).
fof(dt_o_0_0_xboole_0, axiom, v1_xboole_0(o_0_0_xboole_0)).
fof(dt_u1_orders_2, axiom,  (! [A] :  (l1_orders_2(A) => m1_subset_1(u1_orders_2(A), k1_zfmisc_1(k2_zfmisc_1(u1_struct_0(A), u1_struct_0(A))))) ) ).
fof(dt_u1_pre_topc, axiom,  (! [A] :  (l1_pre_topc(A) => m1_subset_1(u1_pre_topc(A), k1_zfmisc_1(k1_zfmisc_1(u1_struct_0(A))))) ) ).
fof(dt_u1_struct_0, axiom, $true).
fof(existence_l1_orders_2, axiom,  (? [A] : l1_orders_2(A)) ).
fof(existence_l1_pre_topc, axiom,  (? [A] : l1_pre_topc(A)) ).
fof(existence_l1_struct_0, axiom,  (? [A] : l1_struct_0(A)) ).
fof(existence_m1_subset_1, axiom,  (! [A] :  (? [B] : m1_subset_1(B, A)) ) ).
fof(free_g1_orders_2, axiom,  (! [A, B] :  (m1_subset_1(B, k1_zfmisc_1(k2_zfmisc_1(A, A))) =>  (! [C, D] :  (g1_orders_2(A, B)=g1_orders_2(C, D) =>  (A=C & B=D) ) ) ) ) ).
fof(t1_subset, axiom,  (! [A] :  (! [B] :  (r2_hidden(A, B) => m1_subset_1(A, B)) ) ) ).
fof(t2_subset, axiom,  (! [A] :  (! [B] :  (m1_subset_1(A, B) =>  (v1_xboole_0(B) | r2_hidden(A, B)) ) ) ) ).
fof(t31_waybel_7, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v2_pre_topc(A) & l1_pre_topc(A)) )  =>  (! [B] :  ( ( ~ (v1_xboole_0(B))  &  (v2_waybel_0(B, k3_yellow_1(u1_struct_0(A))) &  (v13_waybel_0(B, k3_yellow_1(u1_struct_0(A))) &  (v3_waybel_7(B, k3_yellow_1(u1_struct_0(A))) & m1_subset_1(B, k1_zfmisc_1(u1_struct_0(k3_yellow_1(u1_struct_0(A)))))) ) ) )  =>  (! [C] :  (r1_waybel_7(A, B, C) <=> r2_waybel_7(A, B, C)) ) ) ) ) ) ).
fof(t32_waybel_7, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v2_pre_topc(A) & l1_pre_topc(A)) )  =>  (! [B] :  (m1_subset_1(B, u1_struct_0(k2_yellow_1(u1_pre_topc(A)))) =>  (! [C] :  (m1_subset_1(C, u1_struct_0(k2_yellow_1(u1_pre_topc(A)))) =>  (r1_waybel_3(k2_yellow_1(u1_pre_topc(A)), B, C) =>  (! [D] :  ( ( ~ (v1_xboole_0(D))  &  (v1_subset_1(D, u1_struct_0(k3_yellow_1(u1_struct_0(A)))) &  (v2_waybel_0(D, k3_yellow_1(u1_struct_0(A))) &  (v13_waybel_0(D, k3_yellow_1(u1_struct_0(A))) & m1_subset_1(D, k1_zfmisc_1(u1_struct_0(k3_yellow_1(u1_struct_0(A)))))) ) ) )  =>  ~ ( (r2_hidden(B, D) &  (! [E] :  (m1_subset_1(E, u1_struct_0(A)) =>  ~ ( (r2_hidden(E, C) & r1_waybel_7(A, D, E)) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ).
fof(t3_subset, axiom,  (! [A] :  (! [B] :  (m1_subset_1(A, k1_zfmisc_1(B)) <=> r1_tarski(A, B)) ) ) ).
fof(t4_subset, axiom,  (! [A] :  (! [B] :  (! [C] :  ( (r2_hidden(A, B) & m1_subset_1(B, k1_zfmisc_1(C)))  => m1_subset_1(A, C)) ) ) ) ).
fof(t5_subset, axiom,  (! [A] :  (! [B] :  (! [C] :  ~ ( (r2_hidden(A, B) &  (m1_subset_1(B, k1_zfmisc_1(C)) & v1_xboole_0(C)) ) ) ) ) ) ).
fof(t6_boole, axiom,  (! [A] :  (v1_xboole_0(A) => A=k1_xboole_0) ) ).
fof(t7_boole, axiom,  (! [A] :  (! [B] :  ~ ( (r2_hidden(A, B) & v1_xboole_0(B)) ) ) ) ).
fof(t8_boole, axiom,  (! [A] :  (! [B] :  ~ ( (v1_xboole_0(A) &  ( ~ (A=B)  & v1_xboole_0(B)) ) ) ) ) ).
fof(cc1_waybel_7, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v3_orders_2(A) &  (v4_orders_2(A) &  (v5_orders_2(A) & l1_orders_2(A)) ) ) )  =>  (! [B] :  (m1_subset_1(B, k1_zfmisc_1(u1_struct_0(A))) =>  ( ( ~ (v1_xboole_0(B))  &  (v2_waybel_0(B, A) &  (v13_waybel_0(B, A) & v3_waybel_7(B, A)) ) )  =>  ( ~ (v1_xboole_0(B))  &  (v1_subset_1(B, u1_struct_0(A)) &  (v2_waybel_0(B, A) & v13_waybel_0(B, A)) ) ) ) ) ) ) ) ).
fof(fc1_waybel_7, axiom,  (! [A] :  (v1_orders_2(k3_yellow_1(A)) & v11_waybel_1(k3_yellow_1(A))) ) ).
fof(fc2_struct_0, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  & l1_struct_0(A))  =>  ~ (v1_xboole_0(u1_struct_0(A))) ) ) ).
fof(fc2_waybel_7, axiom,  (! [A] :  ( ~ (v1_xboole_0(A))  =>  ( ~ (v7_struct_0(k3_yellow_1(A)))  & v1_orders_2(k3_yellow_1(A))) ) ) ).
fof(fc5_yellow_1, axiom,  (! [A] :  (v1_orders_2(k2_yellow_1(A)) &  (v3_orders_2(k2_yellow_1(A)) &  (v4_orders_2(k2_yellow_1(A)) & v5_orders_2(k2_yellow_1(A))) ) ) ) ).
fof(fc9_yellow_1, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v2_pre_topc(A) & l1_pre_topc(A)) )  =>  ( ~ (v7_struct_0(k2_yellow_1(u1_pre_topc(A))))  &  (v1_orders_2(k2_yellow_1(u1_pre_topc(A))) & v3_lattice3(k2_yellow_1(u1_pre_topc(A)))) ) ) ) ).
fof(rc5_waybel_7, axiom,  (! [A] :  ( ~ (v1_xboole_0(A))  =>  (? [B] :  (m1_subset_1(B, k1_zfmisc_1(u1_struct_0(k3_yellow_1(A)))) &  ( ~ (v1_xboole_0(B))  &  (v2_waybel_0(B, k3_yellow_1(A)) &  (v13_waybel_0(B, k3_yellow_1(A)) & v3_waybel_7(B, k3_yellow_1(A))) ) ) ) ) ) ) ).
