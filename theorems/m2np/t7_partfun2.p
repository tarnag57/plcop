fof(t7_partfun2, conjecture,  (! [A] :  ( ~ (v1_xboole_0(A))  =>  (! [B] :  (m1_subset_1(B, k1_zfmisc_1(A)) =>  (! [C] :  (m1_subset_1(C, A) =>  (! [D] :  ( (v1_funct_1(D) & m1_subset_1(D, k1_zfmisc_1(k2_zfmisc_1(A, A))))  =>  (r2_hidden(C, k9_subset_1(A, k1_relset_1(A, D), B)) => k7_partfun1(A, D, C)=k7_partfun1(A, k1_partfun1(A, A, A, A, k1_partfun2(A, B), D), C)) ) ) ) ) ) ) ) ) ).
fof(commutativity_k3_xboole_0, axiom,  (! [A, B] : k3_xboole_0(A, B)=k3_xboole_0(B, A)) ).
fof(d4_xboole_0, axiom,  (! [A] :  (! [B] :  (! [C] :  (C=k3_xboole_0(A, B) <=>  (! [D] :  (r2_hidden(D, C) <=>  (r2_hidden(D, A) & r2_hidden(D, B)) ) ) ) ) ) ) ).
fof(dt_k1_partfun1, axiom,  (! [A, B, C, D, E, F] :  ( ( (v1_funct_1(E) & m1_subset_1(E, k1_zfmisc_1(k2_zfmisc_1(A, B))))  &  (v1_funct_1(F) & m1_subset_1(F, k1_zfmisc_1(k2_zfmisc_1(C, D)))) )  =>  (v1_funct_1(k1_partfun1(A, B, C, D, E, F)) & m1_subset_1(k1_partfun1(A, B, C, D, E, F), k1_zfmisc_1(k2_zfmisc_1(A, D)))) ) ) ).
fof(dt_k1_partfun2, axiom,  (! [A, B] :  ( ( ~ (v1_xboole_0(A))  & m1_subset_1(B, k1_zfmisc_1(A)))  =>  (v1_funct_1(k1_partfun2(A, B)) & m1_subset_1(k1_partfun2(A, B), k1_zfmisc_1(k2_zfmisc_1(A, A)))) ) ) ).
fof(redefinition_k9_subset_1, axiom,  (! [A, B, C] :  (m1_subset_1(C, k1_zfmisc_1(A)) => k9_subset_1(A, B, C)=k3_xboole_0(B, C)) ) ).
fof(reflexivity_r2_relset_1, axiom,  (! [A, B, C, D] :  ( (m1_subset_1(C, k1_zfmisc_1(k2_zfmisc_1(A, B))) & m1_subset_1(D, k1_zfmisc_1(k2_zfmisc_1(A, B))))  => r2_relset_1(A, B, C, C)) ) ).
fof(t3_partfun2, axiom,  (! [A] :  ( ~ (v1_xboole_0(A))  =>  (! [B] :  ( ~ (v1_xboole_0(B))  =>  (! [C] :  ( ~ (v1_xboole_0(C))  =>  (! [D] :  ( (v1_funct_1(D) & m1_subset_1(D, k1_zfmisc_1(k2_zfmisc_1(C, A))))  =>  (! [E] :  ( (v1_funct_1(E) & m1_subset_1(E, k1_zfmisc_1(k2_zfmisc_1(A, B))))  =>  (! [F] :  ( (v1_funct_1(F) & m1_subset_1(F, k1_zfmisc_1(k2_zfmisc_1(C, B))))  =>  (r2_relset_1(C, B, F, k1_partfun1(C, A, A, B, D, E)) <=>  ( (! [G] :  (m1_subset_1(G, C) =>  (r2_hidden(G, k1_relset_1(C, F)) <=>  (r2_hidden(G, k1_relset_1(C, D)) & r2_hidden(k7_partfun1(A, D, G), k1_relset_1(A, E))) ) ) )  &  (! [G] :  (m1_subset_1(G, C) =>  (r2_hidden(G, k1_relset_1(C, F)) => k7_partfun1(B, F, G)=k7_partfun1(B, E, k7_partfun1(A, D, G))) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ).
fof(t4_subset, axiom,  (! [A] :  (! [B] :  (! [C] :  ( (r2_hidden(A, B) & m1_subset_1(B, k1_zfmisc_1(C)))  => m1_subset_1(A, C)) ) ) ) ).
fof(t6_partfun2, axiom,  (! [A] :  ( ~ (v1_xboole_0(A))  =>  (! [B] :  (m1_subset_1(B, k1_zfmisc_1(A)) =>  (! [C] :  ( (v1_funct_1(C) & m1_subset_1(C, k1_zfmisc_1(k2_zfmisc_1(A, A))))  =>  (r2_relset_1(A, A, C, k1_partfun2(A, B)) <=>  (k1_relset_1(A, C)=B &  (! [D] :  (m1_subset_1(D, A) =>  (r2_hidden(D, B) => k7_partfun1(A, C, D)=D) ) ) ) ) ) ) ) ) ) ) ).