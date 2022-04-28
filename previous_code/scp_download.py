from paramiko import SSHClient
from scp import SCPClient

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect(hostname='longleaf.unc.edu', 
            username='yaoli',
            password='')


# SCPCLient takes a paramiko transport as its only argument
scp = SCPClient(ssh.get_transport())

#scp.put('file_path_on_local_machine', 'file_path_on_remote_machine')

#scp.get('/proj/STOR/yaoli/tangent/results/AA/cifar10/AA_9.npy','C:\\Users\\yaoli\\Documents\\Projects\\tangent\\results\\AA\\cifar10')

d_list = [22507, 38043, 28808, 17271, 10897, 14064, 35448, 28250, 13273, 27521,
        43785, 47944, 19248, 40616, 40496, 33770, 28041, 43146,  7881,  2026,
        48770, 16044,  4851, 40381,  3627, 26892, 12176, 21862, 47859, 30672,
        28115, 46776,  5898, 32665, 10026,  4253, 38628,  4696, 42816, 23974,
        47626, 44694, 45226, 38427, 29642,   559, 10553,  6092, 48502, 22950,
         1564, 40696, 37472,   145, 34539, 22896, 11383, 17336,  9065, 16051,
        33890,  5014, 30104, 36706, 46171, 16393, 10166,  4061, 21985,  1671,
         9373, 48993, 37608, 22254, 48227, 45019,   409,  7619, 33757, 21759,
         3787, 30043, 48232, 31372, 21401, 20510, 33092, 31311, 45277, 15170,
        18962, 34920, 42743,  6941, 29931,  7871, 25852, 31290, 36177, 26616]

for num in d_list:
    file_name = 'A_'+str(num)+'.npy'
    target_path = '/pine/scr/y/a/yaoli/data/A/cifar10/'+file_name
    scp.get(target_path,'D:\\yaoli\\data\\A\\cifar10')



