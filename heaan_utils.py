import piheaan as heaan
from piheaan.math import sort
from piheaan.math import approx # for piheaan math function
import os
import math


class Heaan:
  def __init__(self) -> None:
      # set parameter
      self.params = heaan.ParameterPreset.FGb
      # context has paramter information
      self.context = heaan.make_context(self.params)
      self.key_file_path = "./keys"
      self.eval = None
      self.dec = None
      self.enc = None
      self.sk = None
      self.pk = None
      self.log_slots = 15
      self.num_slots = 2**self.log_slots


  def heaan_initilize(self):

      heaan.make_bootstrappable(self.context) # make parameter bootstrapable

      # # create and save secret keys
      # self.sk = heaan.SecretKey(self.context) # create secret key

      # # create and save public keys
      # key_generator = heaan.KeyGenerator(self.context, self.sk) # create public key
      # key_generator.gen_common_keys()
      # key_generator.save(self.key_file_path+"/") # save public key
      
      # load secret key and public key
      # When a key is created, it can be used again to save a new key without creating a new one
      self.sk = heaan.SecretKey(self.context, self.key_file_path+"/secretkey.bin") # load secret key
      self.pk = heaan.KeyPack(self.context, self.key_file_path+"/") # load public key
      self.pk.load_enc_key()
      self.pk.load_mult_key()

      self.eval = heaan.HomEvaluator(self.context, self.pk) # to load piheaan basic function
      self.dec = heaan.Decryptor(self.context) # for self.decrypt
      self.enc = heaan.Encryptor(self.context) # for self.encrypt

      ctxt1 = heaan.Ciphertext(self.context)
      ctxt2 = heaan.Ciphertext(self.context)

      return ctxt1, ctxt2
      

  def encrypt(self, msg, ctxt):
      self.enc.encrypt(msg, self.pk, ctxt)


  def decrypt(self, msg, ctxt):
      self.dec.decrypt(msg, self.sk, ctxt)


  def similarity_calc(self, res_ctxt):
      sim = heaan.Message(self.log_slots)
      self.dec.decrypt(res_ctxt, self.sk, sim)
      sim_ = sum(sim)/len(sim)
      return sim_


  def feat_msg_generate(self, feat):
      feat_list = feat.tolist()
      feat_padding = feat_list + (self.num_slots-len(feat_list))*[0]
      msg = heaan.Message(self.log_slots)
      for i in range(self.num_slots):
          msg[i] = feat_padding[i]

      return msg


  def cosin_sim(self, ctxt1, ctxt2):

      # # denominator
      # ctxt1 = heaan.Ciphertext(self.context)
      # ctxt1.load(ctxt_path)
      # 안되면 save

      # mult 
      ctxt3 = heaan.Ciphertext(self.context)
      self.eval.mult(ctxt1, ctxt2, ctxt3)

      # sigma
      denom_ctxt = heaan.Ciphertext(self.context)
      self.eval.left_rotate_reduce(ctxt3,1,self.num_slots,denom_ctxt)

      # numerator

      # square
      ctxt1_sqr = heaan.Ciphertext(self.context)
      self.eval.square(ctxt1, ctxt1_sqr)

      ctxt2_sqr = heaan.Ciphertext(self.context)
      self.eval.square(ctxt2, ctxt2_sqr)

      # sigma
      ctxt1_rot = heaan.Ciphertext(self.context)
      self.eval.left_rotate_reduce(ctxt1_sqr,1,self.num_slots,ctxt1_rot)

      ctxt2_rot = heaan.Ciphertext(self.context)
      self.eval.left_rotate_reduce(ctxt2_sqr,1,self.num_slots,ctxt2_rot)

      # sqrt
      ## sigma output range : about 10 ~ 30
      ## divide by 100 and mult 10 to later result value
      ## input range : 2^-18 ≤ x ≤ 2

      hun_msg = heaan.Message(self.log_slots)
      for i in range(self.num_slots):
          hun_msg[i] = 0.01

      self.eval.mult(ctxt1_rot,hun_msg,ctxt1_rot)

      self.eval.mult(ctxt2_rot,hun_msg,ctxt2_rot)

      ctxt1_sqrt = heaan.Ciphertext(self.context)
      approx.sqrt(self.eval,ctxt1_rot,ctxt1_sqrt)

      ctxt2_sqrt = heaan.Ciphertext(self.context)
      approx.sqrt(self.eval,ctxt2_rot,ctxt2_sqrt)

      # mult and inverse 

      ## inverse range : 1 ≤ x ≤ 2^22 or 2^-10 ≤ x ≤ 1
      num_ctxt = heaan.Ciphertext(self.context)
      self.eval.mult(ctxt1_sqrt, ctxt2_sqrt, num_ctxt)

      self.eval.mult(num_ctxt,1000,num_ctxt)

      num_inverse = heaan.Ciphertext(self.context)
      approx.inverse(self.eval,num_ctxt,num_inverse)

      self.eval.mult(num_inverse,10, num_inverse)

      self.eval.bootstrap(num_inverse, num_inverse)

      # cosine similarity
      # mult denominator & numberator^-1
      res_ctxt = heaan.Ciphertext(self.context)
      self.eval.mult(num_inverse,denom_ctxt,res_ctxt)

      return res_ctxt


  def euclidean_distance(self, ctxt1, ctxt2):

      # # sub
      # ctxt1 = heaan.Ciphertext(self.context)
      # ctxt1.load(ctxt_path)

      ctxt3 = heaan.Ciphertext(self.context)
      self.eval.sub(ctxt1, ctxt2, ctxt3)

      # square
      ctxt_square = heaan.Ciphertext(self.context)
      self.eval.square(ctxt3, ctxt_square)

      # sigma
      ctxt_sig = heaan.Ciphertext(self.context)
      self.eval.left_rotate_reduce(ctxt_square,1, self.num_slots, ctxt_sig)

      # sqrt
      ## ctxt_sig is bigger than 2
      ## input range : 2^-18 ≤ x ≤ 2

      self.eval.mult(ctxt_sig,0.01,ctxt_sig)
      ctxt_sqrt = heaan.Ciphertext(self.context)
      approx.sqrt(self.eval,ctxt_sig,ctxt_sqrt)
      self.eval.mult(ctxt_sqrt,10,ctxt_sqrt)

      return ctxt_sqrt


  def manhattan_distance(self, ctxt1, ctxt2):
    
      small_tmp_ctxt= heaan.Ciphertext(self.context)
      small_ctxt = heaan.Ciphertext(self.context)
      big_tmp_ctxt = heaan.Ciphertext(self.context)
      big_ctxt = heaan.Ciphertext(self.context)
      abs_ctxt = heaan.Ciphertext(self.context)
      res_ctxt = heaan.Ciphertext(self.context)
      ctxt3 = heaan.Ciphertext(self.context)
      # ctxt1 = heaan.Ciphertext(self.context)
      # ctxt1.load(ctxt_path)

      ## if ctxt1 < ctxt2 -> 0
      comp_ctxt = heaan.Ciphertext(self.context)
      approx.compare(self.eval, ctxt1, ctxt2, comp_ctxt)

      ## discrete equal zero 
      ## input range : |x| ≤ 54 (x : int)
      discrete_ctxt = heaan.Ciphertext(self.context)
      two_msg = heaan.Message(self.log_slots)
      for i in range(self.num_slots):
          two_msg[i] = 2
      two_ctxt = heaan.Ciphertext(self.context)
      self.enc.encrypt(two_msg,self.pk,two_ctxt)

      comp_tmp_ctxt = heaan.Ciphertext(self.context)
      self.eval.mult(two_ctxt,comp_ctxt,comp_tmp_ctxt)
      approx.discrete_equal_zero(self.eval, comp_tmp_ctxt, discrete_ctxt)

      # sub
      self.eval.sub(ctxt1, ctxt2, ctxt3)

      # small_tmp_ctxt = remain only minus values
      self.eval.mult(ctxt3,discrete_ctxt,small_tmp_ctxt)
      # small_ctxt = - to +
      self.eval.negate(small_tmp_ctxt,small_ctxt)

      one_msg = heaan.Message(self.log_slots)
      for i in range(self.num_slots):
          one_msg[i] = 1
      one_ctxt = heaan.Ciphertext(self.context)
      self.enc.encrypt(one_msg, self.pk, one_ctxt)

      self.eval.sub(one_ctxt,discrete_ctxt,big_tmp_ctxt)
      self.eval.mult(big_tmp_ctxt,ctxt3,big_ctxt)
      self.eval.add(big_ctxt,small_ctxt,abs_ctxt)

      ## sigma
      self.eval.left_rotate_reduce(abs_ctxt,1,self.num_slots,res_ctxt)

      return res_ctxt


  def compare(self, type, thres, comp_ctxt):
    thres_list = []
    thres_list.append(thres)

    thres_list += (self.num_slots-len(thres_list))*[0]

    thres_msg = heaan.Message(self.log_slots)
    for i in range(self.num_slots):
      thres_msg[i] = thres_list[i]

    sub_ctxt = heaan.Ciphertext(self.context)
    if type == 'cosine':
      self.eval.sub(comp_ctxt,thres_msg,sub_ctxt)

    elif type == 'euclidean' or 'manhattan':
      thres_ctxt = heaan.Ciphertext(self.context)
      self.enc.encrypt(thres_msg, self.pk, thres_ctxt)
      self.eval.sub(thres_ctxt,comp_ctxt,sub_ctxt)
      
    ## cos_similarity - threshold > 0 ==> 1
    sign_ctxt = heaan.Ciphertext(self.context)
    approx.sign(self.eval, sub_ctxt, sign_ctxt)

    res = heaan.Message(self.log_slots)
    self.dec.decrypt(sign_ctxt, self.sk, res)

    real = res[0].real
    if -0.0001 < 1-real < 0.0001: res = 'unlock'
    else: res = 'lock'

    return res