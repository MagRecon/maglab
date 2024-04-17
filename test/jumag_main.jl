using JuMag
using NPZ
using Printf

N = 11
nx,ny,nz = N, N ,N

mesh = FDMesh(nx=nx,ny=ny,nz=nz,dx=1e-9,dy=1e-9,dz=1e-9)

function gen_energy(fun)
    sim = Sim(mesh)
    fun(sim, 8e5)
    exch = add_exch(sim, 1e-12)
    dmi = add_dmi(sim, 1e-4)
    anis = add_anis(sim, 1e3)
    demag = add_demag(sim)
    zeeman = add_zeeman(sim, (0,0,1e3))
    interfacial_dmi = add_dmi(sim, 2e-4, type="interfacial")

    m0 = npzread("m0.npy")
    init_m0(sim, reshape(m0, :))

    num = length(sim.interactions)
    all_energy = zeros(num,nx,ny,nz)
    for (i, interaction) in enumerate(sim.interactions)
        JuMag.effective_field(interaction, sim, sim.spin, 0.0)
        all_energy[i,:,:,:] = reshape(interaction.energy, (nx,ny,nz))
    end
    return all_energy, reshape(sim.spin, (3,nx,ny,nz))
end

e, m = gen_energy(set_Ms)
npzwrite("energy.npy", e)

e, m = gen_energy(set_Ms_cylindrical)
npzwrite("energy_cyd.npy", e)
npzwrite("m0_cyd.npy", m)