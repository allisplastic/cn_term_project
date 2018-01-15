import numpy as np
import time
import nest
import sys

t_start = time.time()

rank = nest.Rank()

par     = __import__(sys.argv[1].rpartition("/")[-1].partition(".")[0])
seed    = int(sys.argv[2])

direc   = sys.argv[1].rpartition('/')[0]+"/data/" 

grng_seed               = seed
rng_seeds               = range(seed+1,seed+par.total_num_virtual_procs+1)
numpy_seed              = seed+1000

np.save(direc+"global_seeds.npy",grng_seed)
np.save(direc+"thread_seeds.npy",rng_seeds)
np.save(direc+"numpy_seed.npy",numpy_seed)

np.random.seed(numpy_seed)

# Scaling of time
sim_steps       = np.arange(1,1+par.cicles,1)*par.pre_step
growth_time     = sim_steps[-1]

np.save(direc+"sim_steps.npy",sim_steps)


nest.ResetKernel()

nest.EnableStructuralPlasticity()

nest.SetKernelStatus({"resolution": par.dt, "print_time": False})

nest.SetKernelStatus({
    'structural_plasticity_update_interval' : int(par.MSP_update_interval/par.dt),               # update interval for MSP in time steps
    'total_num_virtual_procs'       : par.total_num_virtual_procs,
    'grng_seed'                     : grng_seed,
    'rng_seeds'                     : rng_seeds,
})

nest.SetDefaults(par.neuron_model, par.neuron_params)

# Create generic neuron with Axon and Dendrite
nest.CopyModel(par.neuron_model, 'excitatory')
nest.CopyModel(par.neuron_model, 'inhibitory')

# growth curves
gc_den = {'growth_curve': par.growth_curve_d, 'z': par.z0_mean, 'growth_rate': par.slope*par.eps, 'eps': par.eps,
          'continuous': False,'tau_vacant':par.tau_vacant}
gc_axon = {'growth_curve': par.growth_curve_a, 'z': par.z0_mean, 'growth_rate': par.slope*par.eps, 'eps': par.eps,
           'continuous': False,'tau_vacant':par.tau_vacant}

nest.SetDefaults('excitatory', 'synaptic_elements', {'Axon_exc': gc_axon, 'Den_exc': gc_den})

# Create synapse models
nest.CopyModel(par.synapse_model, 'msp_excitatory', {"weight":par.weight, "delay":par.delay})

# Use SetKernelStatus to activate the synapse model
nest.SetKernelStatus({
    'structural_plasticity_synapses': {
        'syn1': {
            'model': 'msp_excitatory',
            'post_synaptic_element': 'Den_exc',
            'pre_synaptic_element': 'Axon_exc',
        }
    },
    'autapses': False,
})


# build network
pop_exc = nest.Create('excitatory', par.NE)
pop_inh = nest.Create('inhibitory', par.NI)

for neuron in pop_exc:
    gc_den = {'growth_curve': par.growth_curve_d, 'z': np.random.normal(par.z0_mean,par.z0_std), 'growth_rate': par.slope*par.eps, 'eps': par.eps,
          'continuous': False,'tau_vacant':par.tau_vacant}
    gc_axon = {'growth_curve': par.growth_curve_a, 'z': np.random.normal(par.z0_mean,par.z0_std), 'growth_rate': par.slope*par.eps, 'eps': par.eps,
           'continuous': False,'tau_vacant':par.tau_vacant}
    nest.SetStatus([neuron], 'synaptic_elements', {'Axon_exc': gc_axon, 'Den_exc': gc_den})

nest.CopyModel("static_synapse","device",{"weight":par.weight, "delay":par.delay})

poisson_generator_inh = nest.Create('poisson_generator')
nest.SetStatus(poisson_generator_inh, {"rate": par.rate})
nest.Connect(poisson_generator_inh, pop_inh,'all_to_all',model="device")

poisson_generator_ex = nest.Create('poisson_generator')
nest.SetStatus(poisson_generator_ex, {"rate": par.rate,"stop": float(growth_time)})

nest.Connect(poisson_generator_ex, pop_exc,'all_to_all', model="device")

spike_detector = nest.Create("spike_detector")
nest.SetStatus(spike_detector,{
                                "withtime"  : True,
                                "withgid"   : True,
                                })


nest.Connect(pop_exc+pop_inh, spike_detector,'all_to_all',model="device")

nest.CopyModel("static_synapse","inhibitory_synapse",{"weight":-par.g*par.weight, "delay":par.delay})
source = np.random.random_integers(par.NE+1,par.N_neurons,(par.N_neurons,par.CI))
for n in np.arange(par.N_neurons):
    nest.Connect(list(source[n,:]),[n+1],'all_to_all',model='inhibitory_synapse')

nest.CopyModel("static_synapse","EI_synapse",{"weight":par.weight, "delay":par.delay})
source = np.random.random_integers(1,par.NE,(par.NI,par.CE))
for n in np.arange(par.NI):
    nest.Connect(list(source[n,:]),[par.NE+n+1],'all_to_all',model='EI_synapse')

def simulate_cicle(growth_steps):
    growth_step = growth_steps[1]-growth_steps[0]
    for simulation_time in growth_steps:

        nest.SetStatus(spike_detector,{"start": simulation_time+growth_step-20000.,"stop": simulation_time+growth_step})
        nest.Simulate(growth_step)

        local_connections = nest.GetConnections(pop_exc, pop_exc)
        sources = nest.GetStatus(local_connections,'source')
        targets = nest.GetStatus(local_connections,'target')

        events = nest.GetStatus(spike_detector,'events')[0]
        times = events['times']
        senders = events['senders']

        extension = str(simulation_time+growth_step)+"_"+str(rank)+".npy"
        np.save(direc+"times_"+extension,times)
        np.save(direc+"senders_"+extension,senders)
        nest.SetStatus(spike_detector,'n_events',0)

        del local_connections

        np.save(direc+"sources_"+extension,sources)
        np.save(direc+"targets_"+extension,targets)

# Grow network
growth_steps = np.arange(0, growth_time,par.pre_step)
simulate_cicle(growth_steps)
