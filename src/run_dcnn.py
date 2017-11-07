from dcnn import *
from sklearn.metrics import roc_auc_score

## prepare the dataset 
batch_size=20

train_set = ChestXray_Dataset(use='train',transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)
validation_set=ChestXray_Dataset(use='validation',transform=transform)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)
test_set=ChestXray_Dataset(use='test',transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)
bbox_set=ChestXray_Dataset(use='bboxtest',transform=transform)
bbox_loader = DataLoader(bbox_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)

#model = MyAlexNet().cuda()
model = MyResNet50().cuda()
model =  nn.DataParallel(model)
#criterion = nn.BCEWithLogitsLoss()
criterion = W_BCEWithLogitsLoss()
#optimizer = optim.SGD(list(model.features[-2].parameters())+list(model.classifier.parameters()), lr=0.001, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
bestloss = np.inf                  
for epoch in range(10):                  
    train2(train_loader, model, criterion, optimizer, epoch, 20)
    loss_val = validate(validation_loader, model, criterion)
    isbest = loss_val < bestloss
    bestloss = min(bestloss,loss_val)
    save_checkpoint({
                      'epoch': epoch + 1,
                      'state_dict': model.state_dict(),
                      'loss_val': loss_val,
                      'best_loss': bestloss,
                      'optimizer' : optimizer.state_dict(),
                    }, isbest, filename = 'checkpoint_res50.pth.tar')


y_true,y_score=test(test_loader,model)
roc=[]
for i in range(8):
    roc.append(roc_auc_score(y_true[:,i], y_score[:,i]))