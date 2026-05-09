import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet

def test_data_process():
    test_data = FashionMNIST(
    root='../data',
    train=False,
    transform=transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor()
    ]),
    download=False)

    test_dataloader=Data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    return test_dataloader

def test_model_process(model,test_dataloader):
    device="cuda" if torch.cuda.is_available() else 'cpu'
    model =model.to(device)
    test_corrects=0.0
    test_num=0

    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x=test_data_x.to(device)
            test_data_y=test_data_y.to(device)
            model.eval()
            output=model(test_data_x)
            pre_lab=torch.argmax(output,dim=1)
            test_corrects+=torch.sum(pre_lab==test_data_y.data)
            test_num+=test_data_x.size(0)
    
    test_acc=test_corrects.double().item()/test_num
    print("测试的准确率为:",test_acc)

if __name__=="__main__":
    model=AlexNet()
    model.load_state_dict(torch.load('best_model.pth'))
    test_dataloader=test_data_process()
    print("开始测试")
    test_model_process(model,test_dataloader)

    # device="cuda"if torch.cuda.is_available() else 'cpu'
    # model=model.to(device)

    # classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # with torch.no_grad():
    #     for b_x,b_y in test_dataloader:
    #         b_x=b_x.to(device)
    #         b_y=b_y.to(device)
    #         model.eval()
    #         output=model(b_x)
    #         #这个dim=1什么意思
    #         pre_lab=torch.argmax(output,dim=1)
    #         result=pre_lab.item()
    #         label=b_y.item()
    #         print("预测值",classes[result],"--------","真实值:",classes[label])
