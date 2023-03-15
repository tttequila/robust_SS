import time
from utils import *
from criterion import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
# import soundfile as sf

MAX_INT16 = np.iinfo(np.int16).max

def write_wav(fname, samps, sampling_rate=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    samps = np.divide(samps, np.max(np.abs(samps)))

    # same as MATLAB and kaldi
    if normalize:
        samps = samps * MAX_INT16
        samps = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wavfile.write(fname, sampling_rate, samps)

class Solver(object):
    def __init__(self, train_data, validation_data, test_data, model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.args = args
        self.ae_loss = nn.CrossEntropyLoss()

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter('logs/%s/tensorboard/' % args.log_name)

        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
            
        self._reset()

    def _reset(self):
        self.halving = False
        if self.args.continue_from:
            checkpoint = torch.load('%s/model_dict_best.pt' % self.args.continue_from, map_location='cpu')

            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.start_epoch=checkpoint['epoch']
            self.prev_val_loss = checkpoint['prev_val_loss']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']

            if self.print: print("Resume training from epoch: {}".format(self.start_epoch))
            
        else:
            self.prev_val_loss = float("inf")
            self.best_val_loss = float("inf")
            self.val_no_impv = 0
            self.start_epoch=1
            if self.print: print('Start new training')

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs+1):
            self.joint_loss_weight=epoch
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)

            # Train
            self.model.train()
            start = time.time()
            tr_loss = self._run_one_epoch(data_loader = self.train_data, state='train')
            reduced_tr_loss = self._reduce_tensor(tr_loss)

            if self.print: print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Train Loss {2:.3f}'.format(
                        epoch, time.time() - start, reduced_tr_loss))

            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(data_loader = self.validation_data, state='val')
                reduced_val_loss = self._reduce_tensor(val_loss)

            if self.print: print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Valid Loss {2:.3f}'.format(
                          epoch, time.time() - start, reduced_val_loss))

            # Check whether to adjust learning rate and early stop
            if reduced_val_loss >= self.prev_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 3:
                    self.halving = True
                if self.val_no_impv >= 6:
                    if self.print: print("No imporvement for 6 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0

            # Halfing the learning rate
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr']/2
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = reduced_val_loss

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar('Train loss', reduced_tr_loss, epoch)
                    self.writer.add_scalar('Validation loss', reduced_val_loss, epoch)
                
                # Save regular model
                if epoch %10 == 0:
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_"+str(epoch)+".pt")

                # Save best model
                if reduced_val_loss < self.best_val_loss:
                    self.best_val_loss = reduced_val_loss
                    checkpoint = {'model': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'epoch': epoch+1,
                                    'prev_val_loss': self.prev_val_loss,
                                    'best_val_loss': self.best_val_loss,
                                    'val_no_impv': self.val_no_impv}
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_best.pt")
                    print("Fund new best model, dict saved")

        # Testing
        self.model.eval()
        start = time.time()
        with torch.no_grad():
            test_loss = self._run_one_epoch(data_loader = self.test_data, state='test')
            reduced_test_loss = self._reduce_tensor(test_loss)

        if self.print: print('Test Summary | Test Loss {:.3f}'.format(reduced_test_loss))


    def _run_one_epoch(self, data_loader, state):
        step = 0
        total_loss = 0
        for i, (a_mix, a_tgt, v_tgt, speaker) in enumerate(data_loader):
            a_mix = a_mix.cuda().squeeze().float()
            a_tgt = a_tgt.cuda().squeeze().float()
            v_tgt = v_tgt.cuda().squeeze().float()
            speaker = speaker.cuda().squeeze()

            est_a_tgt = self.model(a_mix, v_tgt)
            max_snr = cal_SISNR(a_tgt, est_a_tgt)
            loss = 0 - torch.mean(max_snr)
            # if self.print: print("loss:", loss.data)

            if state == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.optimizer.step()

            total_loss += loss.data
            
            step += 1
            if step % 200 == 0:
                if self.print: print('step:{} avg loss:{:.3f}'.format(step, total_loss / step))
        
        return total_loss / (i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt

    def cal_acc(self, output, target):
        pred = output.argmax(dim=1, keepdim=False)
        correct = 0
        total = 0
        for i in range(target.shape[0]):
            total += 1
            if (pred[i] == target[i]):
                correct += 1
        return correct/total
