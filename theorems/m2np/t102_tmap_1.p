fof(t102_tmap_1, conjecture,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v2_pre_topc(A) & l1_pre_topc(A)) )  =>  (! [B] :  ( ( ~ (v2_struct_0(B))  & m1_pre_topc(B, A))  =>  (u1_struct_0(k8_tmap_1(A, B))=u1_struct_0(A) &  (! [C] :  (m1_subset_1(C, k1_zfmisc_1(u1_struct_0(A))) =>  (C=u1_struct_0(B) => u1_pre_topc(k8_tmap_1(A, B))=k5_tmap_1(A, C)) ) ) ) ) ) ) ) ).
fof(d10_tmap_1, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v2_pre_topc(A) & l1_pre_topc(A)) )  =>  (! [B] :  (m1_pre_topc(B, A) =>  (! [C] :  ( (v1_pre_topc(C) &  (v2_pre_topc(C) & l1_pre_topc(C)) )  =>  (C=k8_tmap_1(A, B) <=>  (! [D] :  (m1_subset_1(D, k1_zfmisc_1(u1_struct_0(A))) =>  (D=u1_struct_0(B) => C=k6_tmap_1(A, D)) ) ) ) ) ) ) ) ) ) ).
fof(dt_k8_tmap_1, axiom,  (! [A, B] :  ( ( ( ~ (v2_struct_0(A))  &  (v2_pre_topc(A) & l1_pre_topc(A)) )  & m1_pre_topc(B, A))  =>  (v1_pre_topc(k8_tmap_1(A, B)) &  (v2_pre_topc(k8_tmap_1(A, B)) & l1_pre_topc(k8_tmap_1(A, B))) ) ) ) ).
fof(fc5_tmap_1, axiom,  (! [A, B] :  ( ( ( ~ (v2_struct_0(A))  &  (v2_pre_topc(A) & l1_pre_topc(A)) )  & m1_pre_topc(B, A))  =>  ( ~ (v2_struct_0(k8_tmap_1(A, B)))  &  (v1_pre_topc(k8_tmap_1(A, B)) & v2_pre_topc(k8_tmap_1(A, B))) ) ) ) ).
fof(t1_tsep_1, axiom,  (! [A] :  (l1_pre_topc(A) =>  (! [B] :  (m1_pre_topc(B, A) => m1_subset_1(u1_struct_0(B), k1_zfmisc_1(u1_struct_0(A)))) ) ) ) ).
fof(t93_tmap_1, axiom,  (! [A] :  ( ( ~ (v2_struct_0(A))  &  (v2_pre_topc(A) & l1_pre_topc(A)) )  =>  (! [B] :  (m1_subset_1(B, k1_zfmisc_1(u1_struct_0(A))) =>  (u1_struct_0(k6_tmap_1(A, B))=u1_struct_0(A) & u1_pre_topc(k6_tmap_1(A, B))=k5_tmap_1(A, B)) ) ) ) ) ).