rule allostery_sim:
    input:
        "data/raw/laci.hdf"
    output:
        "data/allostery/d{D}/mutations.csv",
        "data/allostery/d{D}/data.csv"
    run:
        import pandas as pd
        import numpy as np
        from tqdm import tqdm

        rng = np.random.default_rng({1: 2, 2: 4, 3: 5}[int(wildcards.D)])
        num_posiitons = 300
        pos_mean_effect = 1.5 * rng.standard_normal(num_posiitons)
        position = np.array(
            [[p + 1 for n in range(6)] for p in range(num_posiitons)]
        ).flatten()
        substitution = np.array(
            [["A", "B", "C", "D", "E", "F"] for p in range(num_posiitons)]
        ).flatten()

        shift_dir = rng.integers(-1, 2, size=len(position))
        eps_AI_shift = np.array(
            [[x for n in range(6)] for x in pos_mean_effect]
        ).flatten()
        eps_AI_shift = eps_AI_shift * shift_dir + rng.standard_normal(len(eps_AI_shift))
        shift_frame = pd.DataFrame(
            {
                "position": position,
                "substitution": substitution,
                "eps_AI_shift": eps_AI_shift,
            }
        )

        if int(wildcards.D) > 1:
            pos_mean_effect = 1.5 * rng.standard_normal(num_posiitons)
            shift_dir = rng.integers(-1, 2, size=len(position))
            eps_RA_shift = np.array(
                [[x for n in range(6)] for x in pos_mean_effect]
            ).flatten()
            eps_RA_shift = eps_RA_shift * shift_dir + rng.standard_normal(
                len(eps_RA_shift)
            )
            shift_frame["eps_RA_shift"] = eps_RA_shift

        if int(wildcards.D) > 2:
            pos_mean_effect = 1.5 * rng.standard_normal(num_posiitons)
            shift_dir = rng.integers(-1, 2, size=len(position))
            log_KA_shift = np.array(
                [[x for n in range(6)] for x in pos_mean_effect]
            ).flatten()
            log_KA_shift = log_KA_shift * shift_dir + rng.standard_normal(
                len(log_KA_shift)
            )
            shift_frame["log_KA_shift"] = log_KA_shift

        shift_frame.to_csv(output[0])

        # Randomly simuilate 100,000 variants, with different mutational distances
        num_variants = 1000
        rng = np.random.default_rng({1: 2, 2: 5}[int(wildcards.D)])
        df = pd.read_hdf(input[0])
        df = df[df.lacI_amino_mutations >= 0]
        df = df[df.lacI_amino_mutations < 15]
        num_mutations = rng.choice(df.lacI_amino_mutations, size=num_variants)
        variant_frame_1 = pd.DataFrame({"num_mutations": num_mutations})

        # build dataset
        rng = np.random.default_rng({1: 3, 2: 3}[int(wildcards.D)])

        position_list = np.unique(shift_frame.position)
        substitution_list = np.unique(shift_frame.substitution)
        var_eps_AI_list = []
        var_list = [[]]*int(wildcards.D)
        mutation_codes = []
        effects = ["eps_AI_shift", "eps_RA_shift"][:int(wildcards.D)]
        for n in tqdm(variant_frame_1.num_mutations):
            pos_list = np.sort(rng.choice(position_list, size=n, replace=False))
            sub_list = rng.choice(substitution_list, size=n)
            codes_list = []
            shifts = [0]*int(wildcards.D)
            for p, sub in zip(pos_list, sub_list):
                codes_list.append(f"{p}{sub}")
                df = shift_frame[
                    (shift_frame.position == p) & (shift_frame.substitution == sub)
                ]
                if len(df) != 1:
                    print(f"unexpected frame length")
                    display(df)
                for i in range(int(wildcards.D)):
                    shifts[i] += df.iloc[0][effects[i]]
            mutation_codes.append(codes_list)

            for i in range(int(wildcards.D)):
                var_list[i].append(shifts[i])

        variant_frame_1["mutation_codes"] = mutation_codes
        #variant_frame_1["eps_AI_shift"] = var_eps_AI_list
        for i in range(int(wildcards.D)):
            variant_frame_1[effects[i]] = var_list[i]

        # for par, fun in zip(["log_g0", "log_ginf", "log_ec50"], [g0, ginf, ec50]):
        #     variant_frame_1[par] = [
        #         np.log10(fun(delta_eps_AI=x + delta_eps_AI_0))
        #         for x in variant_frame_1.eps_AI_shift
        #     ]


        for par, fun in zip(["log_g0", "log_ginf", "log_ec50"], [g0, ginf, ec50]):

            kwargs = {"delta_eps_AI": variant_frame_1.eps_AI_shift + delta_eps_AI_0}

            if int(wildcards.D) > 1:
                kwargs["delta_eps_RA"] = variant_frame_1.eps_RA_shift + delta_eps_RA_0

            tmp = []
            for n in range(num_variants):
                kkwargs = {}
                for k, v in kwargs.items():
                    kkwargs[k] = v.values[n]
                    print(kkwargs)
                    break
                    
                tmp.append(np.log10(fun(**kwargs)))
                break

            variant_frame_1[par] = tmp
            break

rule allostery:
    input:
        expand("data/allostery/d{D}/data.csv", D=range(1, 2))
