# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from projects import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    import os

    # 이미지들이 있는 폴더 경로 설정
    directory = '/home/public/zio/mie1517/aicity2024_track5_test/images'

    # 해당 디렉토리 내의 모든 jpg 파일들의 경로를 저장할 리스트
    file_list = []

    # 디렉토리를 순회하며 .jpg 파일들의 전체 경로를 리스트에 추가
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            file_list.append(os.path.join(directory, filename))

    # 결과 출력
    print(len(file_list))

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    with open('submission_3x_240325_finetuned.txt', 'a') as file:
        for i, f in enumerate(file_list):
            result = inference_detector(model, f)
            #print(i)
            file_name = f.split("/")[-1][:-4]
            print("file name: ", file_name)
            v_id, f_id = file_name.split('_')[0][1:], file_name.split('_')[1][1:]
            # show the results
            bboxes, labels = show_result_pyplot(
                model,
                f,
                result,
                palette=args.palette,
                score_thr=args.score_thr,
                out_file=f"/home/public/zio/test/Co-DETR/outputs/0324/inference/result_{file_name}.jpg")
            
            #print("file id: ", v_id, f_id)
            # 'submission.txt' 파일을 append 모드로 열기
            for b, label in zip(bboxes, labels):
                x, y, w, h, con = b
                label+=1
                print(f"{v_id},{f_id},{x},{y},{w-x},{h-y},{label},{con}")
                file.write(f"{v_id},{f_id},{x},{y},{w-x},{h-y},{label},{con}")
                file.write(f"\n")



async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
