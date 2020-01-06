import os, sys, math, numpy as np
import tensorflow as tf

# From Euler angles to rotation matrix
def eu2om(eu):
    thr = 1e-10
    c0 = math.cos(eu[0])
    c1 = math.cos(eu[1])
    c2 = math.cos(eu[2])
    s0 = math.sin(eu[0])
    s1 = math.sin(eu[1])
    s2 = math.sin(eu[2])

    q = [[c0*c2-s0*c1*s2, s0*c2+c0*c1*s2, s1*s2], [-c0*s2-s0*c1*c2, -s0*s2+c0*c1*c2, s1*c2], [s0*s1, -c0*s1, c1]]
    for i in range(3):
        for j in range(3):
            if abs(q[i][j]) < thr: q[i][j] = 0
    return q

def eu2om_tf(eu):

    c0 = tf.cos(eu[:,0])
    c1 = tf.cos(eu[:,1])
    c2 = tf.cos(eu[:,2])
    s0 = tf.sin(eu[:,0])
    s1 = tf.sin(eu[:,1])
    s2 = tf.sin(eu[:,2])
    eu_sh = eu.get_shape().as_list()
    q = [[c0 * c2 - s0 * c1 * s2, s0 * c2 + c0 * c1 * s2, s1 * s2],
         [-c0 * s2 - s0 * c1 * c2, -s0 * s2 + c0 * c1 * c2, s1 * c2], [s0 * s1, -c0 * s1, c1]]
    #q = [[tf.subtract(tf.multiply(c0,c2),tf.multiply(tf.multiply(s0,c1),s2)), tf.add(tf.multiply(s0,c2),tf.multiply(tf.multiply(c0,c1),s2)), tf.multiply(s1,s2)], [-tf.multiply(c0,s2)-tf.multiply(tf.multiply(s0,c1),c2), -tf.multiply(s0,s2)+tf.multiply(tf.multiply(c0,c1),c2), tf.multiply(s1,c2)], [tf.multipy(s0,s1), -tf.multiply(c0,s1), c1]]

    q = tf.reshape(q, shape=[eu_sh[0]*3*3])

    q_sh = q.get_shape().as_list()

    z_cons = tf.zeros(q_sh, dtype=tf.float32)

    thr_arr = tf.constant(1e-10, shape=q_sh)

    condition = tf.logical_or(tf.logical_or(tf.less(tf.abs(q), thr_arr), tf.is_nan(q)), tf.is_inf(q))

    q = tf.where(condition, thr_arr, q)


    #q = tf.cond(tf.less(q, thr_arr),z_cons, q)

    #print 'qshape1: ', q.get_shape()

    q = tf.reshape(q, [3, 3, eu_sh[0]])
    #print 'qshape2: ', q.get_shape()

    q = tf.transpose(q, perm=[2,0,1])

    #print 'qshape3: ', q.get_shape()
    return q


## DESCRIPTION OF PROGRAM
# CalcDisorientation calculates the disorientation (degrees) between
# two euler angles given in degrees. The disorientation can
# be thought of as the distance metric for the SO(3) manifold,
# just ike the euclidean distance in cartesian manifold.

# The calculation of disorientation involves the use of the
# symmetries in the rotation. A general implementation
# of all point group symmetry has NOT been done. This program
# only considers cubic symmetry which is the case for the material
# Nickel we are loking at right now...

# euler angles are a series of three successive rotation about the
# Z-X-Z axis. Both eu1 and eu2 are 3x1 vectors...


def sym_values():
    sym = np.zeros(shape=(3,3,24))
    sym[1, 0, 0] = 1
    sym[1, 1, 0] = 1
    sym[2, 2, 0] = 1

    sym[0, 0, 1] = 1
    sym[1, 2, 1] = -1
    sym[2, 1, 1] = 1

    sym[0, 0, 2] = 1
    sym[1, 1, 2] = -1
    sym[2, 2, 2] = -1

    sym[0, 0, 3] = 1
    sym[1, 2, 3] = 1
    sym[2, 1, 3] = -1

    sym[0, 2, 4] = -1
    sym[1, 1, 4] = 1
    sym[2, 0, 4] = 1

    sym[0, 2, 5] = 1
    sym[1, 1, 5] = 1
    sym[2, 0, 5] = -1

    sym[0, 0, 6] = -1
    sym[1, 1, 6] = 1
    sym[2, 2, 6] = -1

    sym[0, 0, 7] = -1
    sym[1, 1, 7] = -1
    sym[2, 2, 7] = 1


    sym[0, 1, 8] = 1
    sym[1, 0, 8] = -1
    sym[2, 2, 8] = 1

    sym[0, 1, 9] = -1
    sym[1, 0, 9] = 1
    sym[2, 2, 9] = 1

    sym[0, 1, 10] = -1
    sym[1, 2, 10] = 1
    sym[2, 0, 10] = -1

    sym[0, 2, 11] = 1
    sym[1, 0, 11] = -1
    sym[2, 1, 11] = -1

    sym[0, 1, 12] = -1
    sym[1, 2, 12] = -1
    sym[2, 0, 12] = 1

    sym[0, 2, 13] = -1
    sym[1, 0, 13] = 1
    sym[2, 1, 13] = -1


    sym[0, 1, 14] = 1
    sym[1, 2, 14] = -1
    sym[2, 0, 14] = -1

    sym[0, 2, 15] = -1
    sym[1, 0, 15] = -1
    sym[2, 1, 15] = 1

    sym[0, 1, 16] = 1
    sym[1, 2, 16] = 1
    sym[2, 0, 16] = 1

    sym[0, 2, 17] = 1
    sym[1, 0, 17] = 1
    sym[2, 1, 17] = 1

    sym[0, 1, 18] = 1
    sym[1, 0, 18] = 1
    sym[2, 2, 18] = -1

    sym[0, 0, 19] = -1
    sym[1, 2, 19] = 1
    sym[2, 1, 19] = 1

    sym[0, 2, 20] = 1
    sym[1, 1, 20] = -1
    sym[2, 0, 20] = 1

    sym[0, 0, 21] = -1
    sym[1, 2, 21] = -1
    sym[2, 1, 21] = -1

    sym[0, 2, 22] = -1
    sym[1, 1, 22] = -1
    sym[2, 0, 22] = -1

    sym[0, 1, 23] = -1
    sym[1, 0, 23] = -1
    sym[2, 2, 23] = -1

    return sym

def compute_disorientation(eu1, eu2, is_degree=True):
    #print eu1, eu2
    eu1 = np.asarray(eu1)
    eu2 = np.asarray(eu2)
    if is_degree:
        eu1 = eu1*math.pi/180.0
    #print 'eu1:', eu1
    om1 = eu2om(eu1)
    #print 'om1:', om1
    if is_degree:
        eu2 = eu2*math.pi/180.0
    #print 'eu2:', eu2
    om2 = eu2om(eu2)
    #print om1
    #print om2
    #print 'om2: ', om2
    sym = sym_values()
    dis = 100.0
    for i in range(24):
        #print dis
        g1 = np.matmul(sym[:,:,i],om1)
        #print sym[:,:,i]
        for j in range(24):
            g2 = np.matmul(sym[:,:,j],om2)
            #print sym[:,:,j]
            g = np.matmul(g1,np.transpose(g2))
            ang = 0.5 * (np.trace(g) - 1.0)

            #print 'in loop:', i, j, ang
            try:
                th = math.acos(ang) * 180.0/math.pi
                if th < dis:
                    dis = th

            except:
                pass
            #print(th)

            g = np.matmul(g2,np.transpose(g1))
            ang = 0.5*(np.trace(g)-1.0)
            #print i, j, ang
            try:
                th = math.acos(ang) * 180.0 / math.pi
                if th < dis:
                    dis = th
            except:
                continue
            #print th

    return dis

def compute_disorientations(eu1s, eu2s, is_degree=True):
    total = eu1s.shape[0]
    diss = []
    for i in range(total):
        eu1 = eu1s[i,:]
        eu2 = eu2s[i,:]
        try:
            #print '\nfirst loop for batch:', i, eu1, eu2
            dis = compute_disorientation(eu1, eu2, is_degree)
            diss.append(dis)
        except:
            continue
    print len(diss)
    return np.mean(diss)


def compute_disorientation_tf(eu1s, eu2s):
    #eu1s = tf.transpose(eu1s)
    #eu2s = tf.transpose(eu2s)

    tf_sh = eu1s.get_shape().as_list()
    #print 'tf_sh: ',tf_sh
    #fact = tf.divide(tf.math.pi, 180.0)
    #eu1s = tf.multiply(eu1s, fact)
    #eu2s = tf.multiply(eu2s, fact)

    #print tf_sh, eu1s.get_shape(), eu2s.get_shape()
    sym = tf.constant(sym_values(), dtype=tf.float32)

    om1s = eu2om_tf(eu1s)
    om2s = eu2om_tf(eu2s)

    #for m in range(tf_sh[1]):
    #eu1 = eu1s[:,m]
    #eu2 = eu2s[:,m]
    #om1 = eu2om_tf(eu1)
    #om2 = eu2om_tf(eu2)
    #print om2

    dis_array = tf.constant(100.0, shape=[tf_sh[0]])


    #print 'dis_array: ', dis_array.get_shape()

    #print 'symm shape1: ', sym.get_shape()
    sym = tf.transpose(sym, perm=[2,0,1])
    #print 'symm shape2: ', sym.get_shape()

    #om1s = tf.transpose(om1s, perm=[1,2,0])
    #om2s = tf.transpose(om2s, perm=[1, 2, 0])
    #print 'om1s: ', om1s.get_shape(), om2s.get_shape()

    ones = tf.ones(shape=[tf_sh[0]])

    delta = tf.constant(0e-1, shape=[tf_sh[0]])


    for i in range(24):
        sym_s = sym[i,:,:]
        sym_s = tf.reshape(sym_s, shape=[1,3, 3])

        sym_s = tf.tile(sym_s, multiples=[tf_sh[0],1,1])
        #print 'sym_s: ', sym_s.get_shape()
        g1 = tf.matmul(sym_s, om1s)

        #print 'g1: ', g1.get_shape()

        #print sym[:,:,i]
        for j in range(24):
            sym_s = sym[j,:, :]
            sym_s = tf.reshape(sym_s, shape=[1, 3, 3])

            sym_s = tf.tile(sym_s, multiples=[tf_sh[0], 1, 1])

            g2 = tf.matmul(sym_s,om2s)
            #print 'g2: ', g2.get_shape()
            #print sym[:,:,j]
            g = tf.matmul(g1,tf.transpose(g2, perm=[0,2,1]))
            #print 'g: ', g.get_shape()

            ang = 0.5*tf.subtract(tf.trace(g), ones)

            cond_g = tf.logical_or(tf.greater_equal(ang, ones), tf.is_inf(ang))

            cond_l = tf.logical_or(tf.is_nan(ang), tf.less_equal(ang, -ones))

            ang_r = tf.where(cond_l, -ones+delta, ang)

            ang_r = tf.where(cond_g, -ones+delta, ang_r)

            th = tf.acos(ang_r) * 180.0/math.pi
            #print 'th: ', th.get_shape()

            condition = tf.less(th, dis_array)

            dis_array = tf.where(condition, th, dis_array)

            g = tf.matmul(g2,tf.transpose(g1, perm=[0,2,1]))

            ang = 0.5 * tf.subtract(tf.trace(g), ones)

            cond_g = tf.logical_or(tf.greater_equal(ang, ones), tf.is_inf(ang))

            cond_l = tf.logical_or(tf.is_nan(ang), tf.less_equal(ang, -ones))

            ang_r = tf.where(cond_l, -ones+delta, ang)

            ang_r = tf.where(cond_g, -ones+delta, ang_r)

            th = tf.acos(ang_r) * 180.0 / math.pi

            condition = tf.less(th, dis_array)

            dis_array = tf.where(condition, th, dis_array)

    #diss.append(dis)
    #diss = tf.stack(diss)
    #print 'diss: ', diss.get_shape(), diss.dtype
    #print 'dis_array: ', dis_array.get_shape()
    return dis_array

#print compute_disorientations(np.asarray([[90., 245., 245.],[90.,230., 230.]]), np.asarray([[45., 135., 245.],[90.,235., 230.]]), is_degree=True)


if __name__ == '__main__':

    angle1 = np.asarray([[90., 245., 245.],[90.,230., 230.]])
    angle1 = angle1*math.pi/180.0
    print 'angle1:',angle1
    eu1s = tf.constant(angle1, dtype=tf.float32)
    angle2 = np.asarray([[45., 135., 245.],[90.,235.,230.]])
    angle2 = angle2*math.pi/180.0
    eu2s = tf.constant(angle2, dtype=tf.float32)
    #eu1s = tf.squeeze(eu1s)
    #eu2s = tf.squeeze(eu2s)


    q = eu2om_tf(eu1s)
    #omg2s = eu2om_tf(eu2s)

    #print eu1s.get_shape(), eu2s.get_shape()
    disr = compute_disorientation_tf(eu1s, eu2s)
    #disr = tf.add(eu2om_tf(eu1s), eu2om_tf(eu2s))
    #disr = tf.add(eu1s, eu2s)
    #disr = eu2om_tf(eu1s)

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        #sess.run(init_op)
        sess.run(init_op)
        res = sess.run([disr])
        print res
