open Autograd

(* Let's try to learn f(x) = ax+b. *)
let () =
  Random.self_init ();

  let rand () = Random.float 20. -. 10. in

  let a = rand () in
  let b = rand () in
  let f x = a *. x +. b in

  Printf.printf "a: %.4f / b: %.4f\n%!" a b;

  let pa = const @@ rand () in
  let pb = const @@ rand () in
  let params = [pa;pb] in

  let net x =
    let open Infix in
    pa * x + pb
  in

  let steps = 1000 in
  let learning_rate = 0.01 in
  for step = 0 to steps - 1 do
    let x = rand () in
    let y = net (const x) in
    let loss = powc (sub y (const (f x))) 2. in

    backward loss;
    Printf.printf "step: %4d | loss: %.2f | a: %.2f (%.2f) | b: %.2f (%.2f)\n%!" step (value loss) (value pa) (grad pa) (value pb) (grad pb);

    let lr = learning_rate *. (1. -. float step /. float steps) in
    List.iter (fun p ->
        p.value <- value p -. lr *. grad p;
        p.grad <- 0.
      ) params
  done;
