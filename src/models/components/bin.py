#id 1
           # loss.backward()
            # optimizer.step()
            
            # self.train_loss.update(loss.item(), n=args.batch_size)
            # if args.rank == 0:
            #     print(
            #         "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            #         "loss: {:.4f}".format(self.train_loss.avg),
            #         "time {:.2f}s".format(time.time() - start_time),
                
        #     start_time = time.time()
        # for param in model.parameters():
        #     param.grad = None
        # return self.train_loss.avg
        
        # with torch.no_grad(): # ????
        #     # for idx, batch in enumerate(loader):
        #     data, target = batch["image"], batch["label"]
        #     data, target = data.cuda(0), target.cuda(0)
        #     with autocast(enabled=False):
        #         logits = self.model_inferer(data)
        #     train_labels_list = decollate_batch(target) ## Optimal to use decollate_batch
        #     train_outputs_list = decollate_batch(logits) ## Optimal to use decollate_batch
        #     train_output_convert = [self.post_pred(self.post_sigmoid(val_pred_tensor)) for val_pred_tensor in train_outputs_list]
        #     self.dice_acc.reset()
        #     self.dice_acc(y_pred=train_output_convert, y=train_labels_list)
        #     acc, not_nans = self.dice_acc.aggregate()
        #     acc = acc.cuda(0)

        #     self.train_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            
        #     Dice_TC = self.train_acc.avg[0]
        #     Dice_WT = self.train_acc.avg[1]
        #     Dice_ET = self.train_acc.avg[2]

        #     # loss = self.criterion(logits, target)
            
        #     # self.val_loss.update(loss, data.size(0))
        #     # self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        #     # self.log("val/acc", np.mean(self.val_acc.avg) , on_step=False, on_epoch=True, prog_bar=False, logger=False) ##Mean Val Dice
        #     # self.log("val/acc_best", np.mean(self.val_acc.avg), sync_dist=True, prog_bar=True)
        #     self.log("train/Dice_TC", Dice_TC, on_step=False, on_epoch=True, prog_bar=True)
        #     self.log("train/Dice_WT", Dice_WT, on_step=False, on_epoch=True, prog_bar=True)
        #     self.log("train/Dice_ET", Dice_ET, on_step=False, on_epoch=True, prog_bar=True)
        #     print("Dice_Train_Mean: {:.6f}".format(np.mean(self.train_acc.avg)))
        
# id 2
        # loss, logits, targets = self.model_step(batch)

        # update and log metrics
        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # self.net.eval()
        # start_time = time.time()
        # run_acc = AverageMeter()
    
# id 3 
            # Dice_TC = self.val_acc.avg[0]
            # Dice_WT = self.val_acc.avg[1]
            # Dice_ET = self.val_acc.avg[2]
            # # self.log("val/acc", np.mean(self.val_acc.avg) , on_step=False, on_epoch=True, prog_bar=False, logger=False) ##Mean Val Dice
            # # self.log("val/acc_best", np.mean(self.val_acc.avg), sync_dist=True, prog_bar=True)
            # self.log("val/Dice_TC", Dice_TC, on_step=False, on_epoch=True, prog_bar=True)
            # self.log("val/Dice_WT", Dice_WT, on_step=False, on_epoch=True, prog_bar=True)
            # self.log("val/Dice_ET", Dice_ET, on_step=False, on_epoch=True, prog_bar=True)

            # print("Val Dice: Mean: {:.6g}, TC: {:.6f}, WT: {:.6f}, ET: {:.6f}".format(np.mean(self.val_acc.avg), Dice_TC, Dice_WT, Dice_ET))
